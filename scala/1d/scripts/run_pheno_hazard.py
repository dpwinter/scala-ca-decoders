"""
SCALA1D phenomenological-noise first-passage / hazard runner.

Purpose
-------
Generate time-resolved first-passage statistics for studying

    * approach to stationary logical hazard,
    * dependence of the transient on d and p,
    * dependence on the signal reset time t_R,
    * stationary logical failure rate p_L,
    * relation between stationary hazard and mean failure time.

Output
------
The CSV stores sufficient statistics per reset block rather than
individual trajectory failure times.

Parallel safety
---------------
    * worker processes never write to the CSV;
    * only the parent process merges completed batches;
    * CSV access is protected by an OS file lock;
    * updates use atomic file replacement;
    * existing samples are reused;
    * --shots means TARGET TOTAL shots per configuration.

Phenomenological model
----------------------
By default

    p_data = p
    p_meas = p

Reset schedule
--------------
For each requested reset factor alpha,

    t_R = max(1, round(alpha * d))

The default production prescription is

    alpha = 0.35.

Simulation length
-----------------
By default

    rounds = ceil(rounds_factor * d)

This runtime does NOT depend on the reset period, which is important
when comparing reset schedules at fixed physical evolution time.

Stationary-hazard quality control
---------------------------------
The CSV already contains sufficient statistics to estimate

    p_L = N_fail / exposure

after a burn-in.

By default:

    burn-in        = 20 t_R
    QC window 1    = [20, 50) t_R
    QC window 2    = [50, 100) t_R

The two QC-window hazards are compared using

    z = (h1 - h2) / sqrt(sigma1^2 + sigma2^2),

with

    sigma_i ~= sqrt(N_i) / E_i.

These diagnostics are printed while sampling, but they do not alter
the Monte Carlo data or the estimator stored in the CSV.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd


DEFAULT_OUTPUT = Path(
    "data/pheno/pheno.csv"
)

BASE_SEED = 12345


# =============================================================================
# Output schema
# =============================================================================

COLUMNS = [
    "d",
    "p",
    "p_meas",
    "reset_factor",
    "reset_period",
    "rounds",
    "block",
    "t_start",
    "t_end",
    "block_length",
    "shots",
    "at_risk",
    "failures",
    "survivors_end",
    "exposure",
    "sum_failure_t",
    "sum_failure_t2",
]


KEY_COLUMNS = [
    "d",
    "p",
    "p_meas",
    "reset_factor",
    "reset_period",
    "rounds",
    "block",
]


ADDITIVE_COLUMNS = [
    "shots",
    "at_risk",
    "failures",
    "survivors_end",
    "exposure",
    "sum_failure_t",
    "sum_failure_t2",
]


# =============================================================================
# File locking
# =============================================================================

class FileLock:
    """
    Simple advisory file lock for macOS / Linux.

    The lock is placed on a separate .lock file so atomic replacement of the
    CSV does not invalidate the lock.
    """

    def __init__(self, path: Path):
        self.path = Path(
            str(path) + ".lock"
        )
        self.handle = None

    def __enter__(self):
        import fcntl

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.handle = self.path.open(
            "a+"
        )

        fcntl.flock(
            self.handle.fileno(),
            fcntl.LOCK_EX,
        )

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        import fcntl

        if self.handle is not None:
            fcntl.flock(
                self.handle.fileno(),
                fcntl.LOCK_UN,
            )

            self.handle.close()

        self.handle = None


# =============================================================================
# Basic utilities
# =============================================================================

def canonical_float(x):
    return round(
        float(x),
        12,
    )


def deterministic_seed(
    base_seed,
    d,
    p,
    p_meas,
    reset_factor,
    reset_period,
    rounds,
    existing_shots,
):
    """
    Construct a deterministic seed for each incremental Monte Carlo batch.

    Including existing_shots ensures that increasing the target shot count
    produces a statistically independent continuation rather than repeating
    the previous batch.
    """

    text = (
        f"{int(base_seed)}|"
        f"{int(d)}|"
        f"{float(p):.12g}|"
        f"{float(p_meas):.12g}|"
        f"{float(reset_factor):.12g}|"
        f"{int(reset_period)}|"
        f"{int(rounds)}|"
        f"{int(existing_shots)}"
    )

    digest = hashlib.blake2b(
        text.encode("utf-8"),
        digest_size=8,
    ).digest()

    value = int.from_bytes(
        digest,
        byteorder="little",
        signed=False,
    )

    return int(
        value % (2**63 - 1)
    )


def empty_dataframe():
    return pd.DataFrame(
        columns=COLUMNS
    )


# =============================================================================
# CSV compatibility / loading
# =============================================================================

def upgrade_old_dataframe(df):
    """
    Upgrade an older schema that did not contain reset_factor.

    reset_factor is reconstructed as reset_period / d.
    """

    if (
        "reset_factor" not in df.columns
        and "reset_period" in df.columns
        and "d" in df.columns
    ):
        df = df.copy()

        df["reset_factor"] = (
            df["reset_period"].astype(float)
            / df["d"].astype(float)
        )

    return df


def read_csv_safe(path):
    if not path.exists():
        return empty_dataframe()

    try:
        df = pd.read_csv(
            path
        )

    except pd.errors.EmptyDataError:
        return empty_dataframe()

    if df.empty:
        return empty_dataframe()

    df = upgrade_old_dataframe(
        df
    )

    missing = (
        set(COLUMNS)
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"{path} has incompatible schema. "
            f"Missing columns: {sorted(missing)}"
        )

    return df[
        COLUMNS
    ].copy()


# =============================================================================
# Configuration matching
# =============================================================================

def config_mask(
    df,
    d,
    p,
    p_meas,
    reset_factor,
    reset_period,
    rounds,
):
    if df.empty:
        return np.zeros(
            0,
            dtype=bool,
        )

    return (
        (df["d"].astype(int) == int(d))
        & np.isclose(
            df["p"].astype(float),
            float(p),
            rtol=0.0,
            atol=1e-12,
        )
        & np.isclose(
            df["p_meas"].astype(float),
            float(p_meas),
            rtol=0.0,
            atol=1e-12,
        )
        & np.isclose(
            df["reset_factor"].astype(float),
            float(reset_factor),
            rtol=0.0,
            atol=1e-12,
        )
        & (
            df["reset_period"].astype(int)
            == int(reset_period)
        )
        & (
            df["rounds"].astype(int)
            == int(rounds)
        )
    )


def existing_shots(
    output,
    d,
    p,
    p_meas,
    reset_factor,
    reset_period,
    rounds,
):
    """
    Return the total existing shot count for one complete simulation
    configuration.

    Every block row stores the same total number of trajectories.
    We therefore inspect block 0.
    """

    output = Path(
        output
    )

    with FileLock(
        output
    ):

        df = read_csv_safe(
            output
        )

        mask = config_mask(
            df=df,
            d=d,
            p=p,
            p_meas=p_meas,
            reset_factor=reset_factor,
            reset_period=reset_period,
            rounds=rounds,
        )

        sub = df.loc[
            mask
        ]

        if sub.empty:
            return 0

        block0 = sub[
            sub["block"].astype(int)
            == 0
        ]

        if block0.empty:
            raise RuntimeError(
                "Existing CSV is missing block 0 "
                "for a simulation configuration."
            )

        return int(
            block0.iloc[0]["shots"]
        )


def load_configuration(
    output,
    d,
    p,
    p_meas,
    reset_factor,
    reset_period,
    rounds,
):
    """
    Load the accumulated block statistics for one configuration.
    """

    output = Path(
        output
    )

    with FileLock(
        output
    ):

        df = read_csv_safe(
            output
        )

        mask = config_mask(
            df=df,
            d=d,
            p=p,
            p_meas=p_meas,
            reset_factor=reset_factor,
            reset_period=reset_period,
            rounds=rounds,
        )

        sub = (
            df.loc[mask]
            .sort_values("block")
            .reset_index(drop=True)
        )

    return sub


# =============================================================================
# Hazard / quality-control summaries
# =============================================================================

def rate_from_rows(rows):
    """
    Return failures, exposure, rate, and approximate Poisson standard error.
    """

    if rows.empty:
        return {
            "failures": 0,
            "exposure": 0,
            "rate": math.nan,
            "se": math.nan,
        }

    failures = int(
        rows["failures"].sum()
    )

    exposure = int(
        rows["exposure"].sum()
    )

    if exposure <= 0:
        rate = math.nan
        se = math.nan

    else:
        rate = (
            failures
            / exposure
        )

        if failures > 0:
            se = (
                math.sqrt(failures)
                / exposure
            )
        else:
            se = math.nan

    return {
        "failures": failures,
        "exposure": exposure,
        "rate": rate,
        "se": se,
    }


def stationary_summary(
    sub,
    reset_period,
    burn_in_reset,
    qc_split_reset,
    qc_end_reset,
    qc_min_failures,
    qc_z_threshold,
):
    """
    Estimate the stationary logical hazard and monitor residual time drift.

    Main estimator
    --------------
    Uses every complete reset block beginning at or after

        burn_in_reset * t_R.

    QC windows
    ----------
        window 1 = [burn_in_reset, qc_split_reset) * t_R
        window 2 = [qc_split_reset, qc_end_reset) * t_R

    Because the CSV is stored in reset-period blocks, these boundaries are
    exact whenever the requested limits are integer numbers of reset periods.
    """

    if sub.empty:
        return None

    t_r = int(
        reset_period
    )

    burn_round = int(
        burn_in_reset
        * t_r
    )

    split_round = int(
        qc_split_reset
        * t_r
    )

    end_round = int(
        qc_end_reset
        * t_r
    )

    # All complete blocks after the burn-in.
    stationary_rows = sub[
        sub["t_start"].astype(int)
        >= burn_round
    ]

    total = rate_from_rows(
        stationary_rows
    )

    # QC window 1.
    w1_rows = sub[
        (
            sub["t_start"].astype(int)
            >= burn_round
        )
        & (
            sub["t_end"].astype(int)
            <= split_round
        )
    ]

    # QC window 2.
    w2_rows = sub[
        (
            sub["t_start"].astype(int)
            >= split_round
        )
        & (
            sub["t_end"].astype(int)
            <= end_round
        )
    ]

    w1 = rate_from_rows(
        w1_rows
    )

    w2 = rate_from_rows(
        w2_rows
    )

    # Check whether the requested QC windows are actually fully covered.
    max_t_end = int(
        sub["t_end"].max()
    )

    qc_complete = (
        max_t_end
        >= end_round
    )

    # Difference-of-rates z score.
    z = math.nan

    if (
        w1["failures"] > 0
        and w2["failures"] > 0
        and np.isfinite(w1["se"])
        and np.isfinite(w2["se"])
    ):
        denom = math.sqrt(
            w1["se"] ** 2
            + w2["se"] ** 2
        )

        if denom > 0:
            z = (
                w1["rate"]
                - w2["rate"]
            ) / denom

    flags = []

    if not qc_complete:
        flags.append(
            "QC_INCOMPLETE"
        )

    if (
        w1["failures"]
        < qc_min_failures
        or w2["failures"]
        < qc_min_failures
    ):
        flags.append(
            "LOW_STATS"
        )

    if (
        np.isfinite(z)
        and abs(z)
        > qc_z_threshold
    ):
        flags.append(
            "DRIFT?"
        )

    if not flags:
        flags.append(
            "OK"
        )

    return {
        "burn_round": burn_round,
        "split_round": split_round,
        "end_round": end_round,
        "total": total,
        "w1": w1,
        "w2": w2,
        "z": z,
        "flags": flags,
        "qc_complete": qc_complete,
    }


def format_rate(x):
    if not np.isfinite(x):
        return "nan"

    return f"{x:.3e}"


def format_z(x):
    if not np.isfinite(x):
        return "nan"

    return f"{x:+.2f}"


def print_stationary_summary(
    sub,
    reset_period,
    burn_in_reset,
    qc_split_reset,
    qc_end_reset,
    qc_min_failures,
    qc_z_threshold,
):
    summary = stationary_summary(
        sub=sub,
        reset_period=reset_period,
        burn_in_reset=burn_in_reset,
        qc_split_reset=qc_split_reset,
        qc_end_reset=qc_end_reset,
        qc_min_failures=qc_min_failures,
        qc_z_threshold=qc_z_threshold,
    )

    if summary is None:
        return

    total = summary["total"]
    w1 = summary["w1"]
    w2 = summary["w2"]

    print(
        "    stationary: "
        f"burn={burn_in_reset:g} tR  "
        f"Nfail={total['failures']:7d}  "
        f"E={total['exposure']:12d}  "
        f"pL={format_rate(total['rate'])}  "
        f"SE={format_rate(total['se'])}"
    )

    print(
        "    QC: "
        f"[{burn_in_reset:g},{qc_split_reset:g})tR "
        f"N={w1['failures']:6d} "
        f"h={format_rate(w1['rate'])}   "
        f"[{qc_split_reset:g},{qc_end_reset:g})tR "
        f"N={w2['failures']:6d} "
        f"h={format_rate(w2['rate'])}   "
        f"z={format_z(summary['z'])}   "
        f"{' '.join(summary['flags'])}"
    )


# =============================================================================
# Aggregate first-passage times
# =============================================================================

def aggregate_first_failures(
    first_failure,
    d,
    p,
    p_meas,
    reset_factor,
    reset_period,
    rounds,
):
    """
    Convert first-passage samples to reset-block sufficient statistics.

    Block b covers rounds

        t_start + 1, ..., t_end

    with

        t_start = b * reset_period
        t_end   = min((b+1)*reset_period, rounds)

    exposure
    --------
    Total trajectory-rounds at risk within the block.

    A trajectory failing at round T contributes

        T - t_start

    rounds of exposure.

    A trajectory surviving the entire block contributes

        block_length.
    """

    first_failure = np.asarray(
        first_failure,
        dtype=np.int64,
    )

    shots = int(
        len(first_failure)
    )

    n_blocks = int(
        math.ceil(
            rounds
            / reset_period
        )
    )

    rows = []

    for block in range(
        n_blocks
    ):

        t_start = int(
            block
            * reset_period
        )

        t_end = int(
            min(
                (block + 1)
                * reset_period,
                rounds,
            )
        )

        block_length = int(
            t_end
            - t_start
        )

        alive_start = (
            (first_failure < 0)
            | (first_failure > t_start)
        )

        at_risk = int(
            np.count_nonzero(
                alive_start
            )
        )

        if at_risk == 0:
            break

        failed_here = (
            (first_failure > t_start)
            & (first_failure <= t_end)
        )

        failures = int(
            np.count_nonzero(
                failed_here
            )
        )

        survived_end = (
            (first_failure < 0)
            | (first_failure > t_end)
        )

        survivors_end = int(
            np.count_nonzero(
                survived_end
            )
        )

        failure_times = (
            first_failure[
                failed_here
            ]
        )

        exposure_failed = int(
            np.sum(
                failure_times
                - t_start,
                dtype=np.int64,
            )
        )

        exposure_survivors = int(
            survivors_end
            * block_length
        )

        exposure = int(
            exposure_failed
            + exposure_survivors
        )

        if failures > 0:

            sum_failure_t = int(
                np.sum(
                    failure_times,
                    dtype=np.int64,
                )
            )

            failure_times_64 = (
                failure_times.astype(
                    np.int64,
                    copy=False,
                )
            )

            sum_failure_t2 = int(
                np.sum(
                    failure_times_64
                    * failure_times_64,
                    dtype=np.int64,
                )
            )

        else:

            sum_failure_t = 0
            sum_failure_t2 = 0

        rows.append({
            "d": int(d),
            "p": float(p),
            "p_meas": float(p_meas),

            "reset_factor": float(
                reset_factor
            ),

            "reset_period": int(
                reset_period
            ),

            "rounds": int(
                rounds
            ),

            "block": int(
                block
            ),

            "t_start": int(
                t_start
            ),

            "t_end": int(
                t_end
            ),

            "block_length": int(
                block_length
            ),

            "shots": int(
                shots
            ),

            "at_risk": int(
                at_risk
            ),

            "failures": int(
                failures
            ),

            "survivors_end": int(
                survivors_end
            ),

            "exposure": int(
                exposure
            ),

            "sum_failure_t": int(
                sum_failure_t
            ),

            "sum_failure_t2": int(
                sum_failure_t2
            ),
        })

    return pd.DataFrame(
        rows,
        columns=COLUMNS,
    )


# =============================================================================
# Run one Monte Carlo batch
# =============================================================================

def run_batch(task):

    (
        d,
        p,
        p_meas,
        reset_factor,
        reset_period,
        rounds,
        batch_shots,
        seed,
        numba_threads,
    ) = task

    from numba import set_num_threads
    from src.scala1d_optimized import pheno

    set_num_threads(
        int(numba_threads)
    )

    first_failure = pheno(
        d=int(d),
        p=float(p),
        p_meas=float(p_meas),
        shots=int(batch_shots),
        rounds=int(rounds),
        reset_period=int(reset_period),
        seed=int(seed),
    )

    result = aggregate_first_failures(
        first_failure=first_failure,
        d=d,
        p=p,
        p_meas=p_meas,
        reset_factor=reset_factor,
        reset_period=reset_period,
        rounds=rounds,
    )

    n_failed = int(
        np.count_nonzero(
            first_failure > 0
        )
    )

    n_censored = int(
        np.count_nonzero(
            first_failure < 0
        )
    )

    if n_failed > 0:

        mean_failed_time = float(
            np.mean(
                first_failure[
                    first_failure > 0
                ]
            )
        )

    else:

        mean_failed_time = math.nan

    return {
        "d": int(d),
        "p": float(p),
        "p_meas": float(p_meas),

        "reset_factor": float(
            reset_factor
        ),

        "reset_period": int(
            reset_period
        ),

        "rounds": int(
            rounds
        ),

        "batch_shots": int(
            batch_shots
        ),

        "n_failed": int(
            n_failed
        ),

        "n_censored": int(
            n_censored
        ),

        "mean_failed_time": float(
            mean_failed_time
        ),

        "data": result,
    }


# =============================================================================
# Merge completed batch into shared CSV
# =============================================================================

def merge_batch(
    output,
    batch_df,
):
    """
    Add one independent completed Monte Carlo batch to the shared CSV.

    The operation is protected with a lock and committed through atomic
    replacement.
    """

    output = Path(
        output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with FileLock(
        output
    ):

        current = read_csv_safe(
            output
        )

        if current.empty:

            merged = (
                batch_df.copy()
            )

        else:

            combined = pd.concat(
                [
                    current,
                    batch_df,
                ],
                ignore_index=True,
            )

            grouped_rows = []

            for _, sub in combined.groupby(
                KEY_COLUMNS,
                sort=False,
                dropna=False,
            ):

                row = (
                    sub.iloc[-1]
                    .copy()
                )

                row["t_start"] = int(
                    sub.iloc[0]["t_start"]
                )

                row["t_end"] = int(
                    sub.iloc[0]["t_end"]
                )

                row["block_length"] = int(
                    sub.iloc[0]["block_length"]
                )

                for col in ADDITIVE_COLUMNS:

                    row[col] = int(
                        sub[col].sum()
                    )

                grouped_rows.append(
                    row
                )

            merged = pd.DataFrame(
                grouped_rows,
                columns=COLUMNS,
            )

        merged = merged.sort_values(
            [
                "d",
                "p",
                "p_meas",
                "reset_factor",
                "reset_period",
                "rounds",
                "block",
            ]
        ).reset_index(
            drop=True
        )

        fd, temp_name = (
            tempfile.mkstemp(
                prefix=(
                    output.name
                    + "."
                ),
                suffix=".tmp",
                dir=str(
                    output.parent
                ),
            )
        )

        os.close(
            fd
        )

        try:

            merged.to_csv(
                temp_name,
                index=False,
            )

            os.replace(
                temp_name,
                output,
            )

        finally:

            if os.path.exists(
                temp_name
            ):
                os.unlink(
                    temp_name
                )


# =============================================================================
# CLI
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "SCALA1D phenomenological-noise "
            "first-passage / hazard Monte Carlo."
        )
    )

    parser.add_argument(
        "--d",
        nargs="+",
        type=int,
        required=True,
        help="Odd code distances.",
    )

    parser.add_argument(
        "--p",
        nargs="+",
        type=float,
        required=True,
        help=(
            "Data-qubit error probabilities per round."
        ),
    )

    parser.add_argument(
        "--p-meas",
        nargs="*",
        type=float,
        default=None,
        help=(
            "Measurement-error probabilities. "
            "If omitted, p_meas = p."
        ),
    )

    parser.add_argument(
        "--shots",
        type=int,
        default=50_000,
        help=(
            "TARGET TOTAL shots for every "
            "(d,p,p_meas,reset_factor) configuration. "
            "Existing samples are reused."
        ),
    )

    parser.add_argument(
        "--rounds-factor",
        type=float,
        default=120.0,
        help=(
            "Simulation length in units of d: "
            "rounds = ceil(rounds_factor*d). "
            "Default: 120."
        ),
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help=(
            "Explicit number of rounds. "
            "Overrides --rounds-factor."
        ),
    )

    parser.add_argument(
        "--reset-factor",
        nargs="+",
        type=float,
        default=[0.35],
        help=(
            "One or more reset factors alpha with "
            "t_R = round(alpha*d). "
            "Production default: 0.35."
        ),
    )

    parser.add_argument(
        "--reset-period",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Optional explicit reset periods. "
            "If given, --reset-factor is ignored. "
            "These absolute periods are applied to "
            "every requested d."
        ),
    )

    # -----------------------------------------------------------------
    # Stationary-hazard QC
    # -----------------------------------------------------------------

    parser.add_argument(
        "--burn-in-reset",
        type=int,
        default=20,
        help=(
            "Discard this many reset periods before estimating "
            "the stationary logical hazard. Default: 20."
        ),
    )

    parser.add_argument(
        "--qc-split-reset",
        type=int,
        default=50,
        help=(
            "End of the first stationarity QC window, in reset "
            "periods. Default: 50."
        ),
    )

    parser.add_argument(
        "--qc-end-reset",
        type=int,
        default=100,
        help=(
            "End of the second stationarity QC window, in reset "
            "periods. Default: 100."
        ),
    )

    parser.add_argument(
        "--qc-min-failures",
        type=int,
        default=50,
        help=(
            "Minimum failures required in each QC window before "
            "the stationarity comparison is considered well "
            "sampled. Default: 50."
        ),
    )

    parser.add_argument(
        "--qc-z-threshold",
        type=float,
        default=2.0,
        help=(
            "Flag DRIFT? when the two QC-window hazards differ "
            "by more than this many approximate standard errors. "
            "Default: 2."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of independent point processes."
        ),
    )

    parser.add_argument(
        "--numba-threads",
        type=int,
        default=1,
        help=(
            "Numba threads inside each worker."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=BASE_SEED,
        help=(
            "Base random-number seed."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(
            DEFAULT_OUTPUT
        ),
        help=(
            "Shared aggregate CSV. "
            "Default: data/pheno/pheno.csv"
        ),
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():

    args = parse_args()

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    if args.shots <= 0:
        raise ValueError(
            "--shots must be positive."
        )

    if args.workers <= 0:
        raise ValueError(
            "--workers must be positive."
        )

    if args.numba_threads <= 0:
        raise ValueError(
            "--numba-threads must be positive."
        )

    if args.rounds_factor <= 0:
        raise ValueError(
            "--rounds-factor must be positive."
        )

    if args.burn_in_reset < 0:
        raise ValueError(
            "--burn-in-reset must be non-negative."
        )

    if (
        args.qc_split_reset
        <= args.burn_in_reset
    ):
        raise ValueError(
            "--qc-split-reset must exceed --burn-in-reset."
        )

    if (
        args.qc_end_reset
        <= args.qc_split_reset
    ):
        raise ValueError(
            "--qc-end-reset must exceed --qc-split-reset."
        )

    if args.qc_min_failures < 0:
        raise ValueError(
            "--qc-min-failures must be non-negative."
        )

    if args.qc_z_threshold <= 0:
        raise ValueError(
            "--qc-z-threshold must be positive."
        )

    for alpha in args.reset_factor:
        if alpha <= 0:
            raise ValueError(
                "--reset-factor values must be positive."
            )

    if args.reset_period is not None:
        for t_r in args.reset_period:
            if t_r <= 0:
                raise ValueError(
                    "--reset-period values must be positive."
                )

    # -----------------------------------------------------------------
    # Measurement-noise mode
    # -----------------------------------------------------------------

    if (
        args.p_meas is None
        or len(args.p_meas) == 0
    ):
        measurement_mode = "equal"

    else:
        measurement_mode = "explicit"

    tasks = []

    print()
    print("=" * 118)

    print(
        "SCALA1D PHENOMENOLOGICAL "
        "FIRST-PASSAGE / STATIONARY-HAZARD SCAN"
    )

    print("=" * 118)

    # -----------------------------------------------------------------
    # Construct configurations
    # -----------------------------------------------------------------

    for d in args.d:

        if d <= 0:
            raise ValueError(
                f"d must be positive, got {d}."
            )

        if d % 2 == 0:
            raise ValueError(
                f"d must be odd, got {d}."
            )

        if args.rounds is not None:

            rounds = int(
                args.rounds
            )

        else:

            rounds = max(
                1,
                int(
                    math.ceil(
                        args.rounds_factor
                        * d
                    )
                ),
            )

        # -------------------------------------------------------------
        # Reset schedules
        # -------------------------------------------------------------

        reset_configs = []

        if (
            args.reset_period is not None
            and len(args.reset_period) > 0
        ):

            for t_r in args.reset_period:

                reset_configs.append(
                    (
                        float(t_r / d),
                        int(t_r),
                    )
                )

        else:

            for alpha in args.reset_factor:

                alpha = canonical_float(
                    alpha
                )

                t_r = max(
                    1,
                    int(
                        round(
                            alpha
                            * d
                        )
                    ),
                )

                reset_configs.append(
                    (
                        float(alpha),
                        int(t_r),
                    )
                )

        seen_reset = set()
        unique_reset_configs = []

        for alpha, t_r in reset_configs:

            key = (
                canonical_float(alpha),
                int(t_r),
            )

            if key in seen_reset:
                continue

            seen_reset.add(
                key
            )

            unique_reset_configs.append(
                (
                    alpha,
                    t_r,
                )
            )

        for p in args.p:

            p = float(
                p
            )

            if not (
                0.0 <= p <= 1.0
            ):
                raise ValueError(
                    f"Invalid p={p}."
                )

            if measurement_mode == "equal":

                measurement_values = [
                    p
                ]

            else:

                measurement_values = [
                    float(q)
                    for q in args.p_meas
                ]

            for p_meas in measurement_values:

                if not (
                    0.0 <= p_meas <= 1.0
                ):
                    raise ValueError(
                        f"Invalid p_meas={p_meas}."
                    )

                for (
                    reset_factor,
                    reset_period,
                ) in unique_reset_configs:

                    have = existing_shots(
                        output=output,
                        d=d,
                        p=p,
                        p_meas=p_meas,
                        reset_factor=reset_factor,
                        reset_period=reset_period,
                        rounds=rounds,
                    )

                    need = max(
                        0,
                        args.shots
                        - have,
                    )

                    if need == 0:

                        print(
                            f"d={d:3d} "
                            f"p={p:.5f} "
                            f"q={p_meas:.5f} "
                            f"alpha={reset_factor:5.3f} "
                            f"tR={reset_period:4d} "
                            f"T={rounds:7d} "
                            f"N={have:9d} "
                            f"TARGET REACHED"
                        )

                        sub = load_configuration(
                            output=output,
                            d=d,
                            p=p,
                            p_meas=p_meas,
                            reset_factor=reset_factor,
                            reset_period=reset_period,
                            rounds=rounds,
                        )

                        print_stationary_summary(
                            sub=sub,
                            reset_period=reset_period,
                            burn_in_reset=args.burn_in_reset,
                            qc_split_reset=args.qc_split_reset,
                            qc_end_reset=args.qc_end_reset,
                            qc_min_failures=args.qc_min_failures,
                            qc_z_threshold=args.qc_z_threshold,
                        )

                        continue

                    seed = deterministic_seed(
                        base_seed=args.seed,
                        d=d,
                        p=p,
                        p_meas=p_meas,
                        reset_factor=reset_factor,
                        reset_period=reset_period,
                        rounds=rounds,
                        existing_shots=have,
                    )

                    tasks.append(
                        (
                            d,
                            p,
                            p_meas,
                            reset_factor,
                            reset_period,
                            rounds,
                            need,
                            seed,
                            args.numba_threads,
                        )
                    )

                    print(
                        f"d={d:3d} "
                        f"p={p:.5f} "
                        f"q={p_meas:.5f} "
                        f"alpha={reset_factor:5.3f} "
                        f"tR={reset_period:4d} "
                        f"T={rounds:7d} "
                        f"existing={have:9d} "
                        f"adding={need:9d}"
                    )

    if not tasks:

        print()
        print(
            "All requested configurations already "
            "satisfy the target shot count."
        )

        return

    print()
    print("-" * 118)

    print(
        f"points             = {len(tasks)}"
    )

    print(
        f"workers            = {args.workers}"
    )

    print(
        f"numba threads      = {args.numba_threads} / worker"
    )

    print(
        f"target shots       = {args.shots}"
    )

    print(
        f"output             = {output}"
    )

    if args.rounds is None:

        print(
            f"runtime            = "
            f"ceil({args.rounds_factor} * d)"
        )

    else:

        print(
            f"runtime            = {args.rounds}"
        )

    if (
        args.reset_period is not None
        and len(args.reset_period) > 0
    ):

        print(
            "reset periods      = "
            + " ".join(
                str(x)
                for x in args.reset_period
            )
        )

    else:

        print(
            "reset factors      = "
            + " ".join(
                f"{x:g}"
                for x in args.reset_factor
            )
        )

    print(
        f"burn-in            = {args.burn_in_reset} tR"
    )

    print(
        f"QC windows         = "
        f"[{args.burn_in_reset},{args.qc_split_reset}) tR, "
        f"[{args.qc_split_reset},{args.qc_end_reset}) tR"
    )

    print(
        f"QC min failures    = {args.qc_min_failures} / window"
    )

    print(
        f"QC drift threshold = |z| > {args.qc_z_threshold:g}"
    )

    print("-" * 118)
    print()

    # -------------------------------------------------------------------------
    # Parallel Monte Carlo
    # -------------------------------------------------------------------------

    with ProcessPoolExecutor(
        max_workers=args.workers
    ) as pool:

        futures = {
            pool.submit(
                run_batch,
                task,
            ): task

            for task in tasks
        }

        completed = 0

        for future in as_completed(
            futures
        ):

            task = futures[
                future
            ]

            d = task[0]
            p = task[1]
            reset_factor = task[3]

            try:

                result = (
                    future.result()
                )

            except Exception as exc:

                print(
                    f"FAILED: "
                    f"d={d} "
                    f"p={p} "
                    f"alpha={reset_factor}: "
                    f"{exc}"
                )

                raise

            # Commit completed point immediately.
            merge_batch(
                output=output,
                batch_df=result["data"],
            )

            completed += 1

            frac_failed = (
                result["n_failed"]
                / result["batch_shots"]
            )

            print(
                f"[{completed:3d}/{len(tasks):3d}] "
                f"d={result['d']:3d} "
                f"p={result['p']:.5f} "
                f"q={result['p_meas']:.5f} "
                f"alpha={result['reset_factor']:5.3f} "
                f"tR={result['reset_period']:4d} "
                f"T={result['rounds']:7d} "
                f"Nadd={result['batch_shots']:8d} "
                f"failed={result['n_failed']:8d} "
                f"({frac_failed:.5f}) "
                f"censored={result['n_censored']:8d}"
            )

            # -------------------------------------------------------------
            # Re-read accumulated configuration and report stationary p_L
            # plus the stationarity QC diagnostic.
            # -------------------------------------------------------------

            sub = load_configuration(
                output=output,
                d=result["d"],
                p=result["p"],
                p_meas=result["p_meas"],
                reset_factor=result["reset_factor"],
                reset_period=result["reset_period"],
                rounds=result["rounds"],
            )

            print_stationary_summary(
                sub=sub,
                reset_period=result["reset_period"],
                burn_in_reset=args.burn_in_reset,
                qc_split_reset=args.qc_split_reset,
                qc_end_reset=args.qc_end_reset,
                qc_min_failures=args.qc_min_failures,
                qc_z_threshold=args.qc_z_threshold,
            )

    print()
    print("=" * 118)

    print(
        f"Saved: {output}"
    )

    print("=" * 118)


if __name__ == "__main__":
    main()
