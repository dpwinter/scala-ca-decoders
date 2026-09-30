"""
SCALA1D phenomenological p_L scan with optional signal noise.

Definition
----------
p_L = h_inf,

with h_inf estimated from the stationary first-passage hazard.

Fixed protocol for every (d,p,q,p_sig):

    t_R = round(reset_factor * d)
    B   = burn_reset * t_R

    burn-in: [0, B)
    W1:      [B, 2B)
    W2:      [2B, 3B)

Trajectories start clean and are absorbed permanently at their
FIRST logical failure.

For each window:

    h_i = F_i / E_i,

where F_i is the number of first failures and E_i is the total
surviving trajectory-round exposure.

Stationarity diagnostic:

    z = |h1-h2| / sqrt(se1^2 + se2^2)

and

    rel = |h1-h2| / ((h1+h2)/2)

If both windows contain enough failures and satisfy the fixed
stationarity criteria, report

    p_L = (F1+F2)/(E1+E2).

No adaptive window search.
No runtime extension.
No restarting or cloning failed trajectories.

Signal noise
------------
Each stored left/right signal bit is independently flipped with
probability p_sig after the deterministic SCALA update of each CA
step and before a scheduled signal reset. Thus, on reset rounds the
reset clears both accumulated signals and signal-bit errors.

If --p-sig is omitted, p_sig=p for each scanned p value.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from numba import njit, prange, set_num_threads

from src.scala1d_optimized import (
    ALL,
    RNG_SCALE,
    bernoulli64,
    scala_step,
)


BASE_SEED = 12345

DEFAULT_OUTPUT = Path(
    "data/pheno/pheno_pL_signal_noise.csv"
)


# =============================================================================
# Utilities
# =============================================================================

def probability_threshold(p: float) -> int:
    return int(round(float(p) * RNG_SCALE))


def deterministic_seed(base_seed, *items):
    s = "|".join([str(base_seed)] + [str(x) for x in items])

    digest = hashlib.blake2b(
        s.encode(),
        digest_size=8,
    ).digest()

    return (
        int.from_bytes(digest, "little")
        % (2**63 - 1)
    )


def atomic_write_csv(df, path):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )

    os.close(fd)

    try:
        df.to_csv(
            tmp,
            index=False,
        )

        os.replace(
            tmp,
            path,
        )

    finally:

        if os.path.exists(tmp):
            os.unlink(tmp)


# =============================================================================
# Fast uint64 population count
# =============================================================================

@njit(inline="always")
def popcount64(x):

    x = x - (
        (x >> np.uint64(1))
        & np.uint64(0x5555555555555555)
    )

    x = (
        x & np.uint64(0x3333333333333333)
    ) + (
        (x >> np.uint64(2))
        & np.uint64(0x3333333333333333)
    )

    x = (
        x + (x >> np.uint64(4))
    ) & np.uint64(
        0x0F0F0F0F0F0F0F0F
    )

    x = (
        x
        * np.uint64(0x0101010101010101)
    )

    return int(
        x >> np.uint64(56)
    )


# =============================================================================
# Logical-sector test
# =============================================================================

@njit(inline="always")
def majority_mask(
    errors,
    d,
    valid,
    counter,
):
    """
    Return lanes with data-error weight > d/2.
    """

    nbits = counter.shape[0]

    for j in range(nbits):
        counter[j] = np.uint64(0)

    for i in range(d):

        carry = (
            errors[i]
            & valid
        )

        for j in range(nbits):

            new_carry = (
                counter[j]
                & carry
            )

            counter[j] ^= carry
            carry = new_carry

            if carry == 0:
                break

    threshold = (
        d // 2
    )

    equal = valid
    greater = np.uint64(0)

    for j in range(
        nbits - 1,
        -1,
        -1,
    ):

        plane = (
            counter[j]
            & valid
        )

        if (
            (threshold >> j)
            & 1
        ) == 0:

            greater |= (
                equal
                & plane
            )

            equal &= ~plane

        else:
            equal &= plane

        equal &= valid
        greater &= valid

    return (
        greater
        & valid
    )


# =============================================================================
# One phenomenological round
# =============================================================================

@njit(inline="always")
def apply_round(
    errors,
    left,
    right,
    defect,
    new_left,
    new_right,
    d,
    active,
    rng,
    threshold_data,
    threshold_meas,
    threshold_sig,
    reset_now,
):

    # ---------------------------------------------------------------------
    # Data noise
    # ---------------------------------------------------------------------

    if threshold_data > 0:

        for i in range(d):

            rng, noise = bernoulli64(
                rng,
                threshold_data,
            )

            errors[i] ^= (
                noise
                & active
            )


    # ---------------------------------------------------------------------
    # Syndrome + measurement noise
    # ---------------------------------------------------------------------

    for i in range(d):

        im = i - 1

        if im < 0:
            im = d - 1

        if threshold_meas > 0:

            rng, meas = bernoulli64(
                rng,
                threshold_meas,
            )

            meas &= active

        else:

            meas = np.uint64(0)

        defect[i] = (
            errors[im]
            ^ errors[i]
            ^ meas
        )


    # ---------------------------------------------------------------------
    # Deterministic decoder update
    # ---------------------------------------------------------------------

    scala_step(
        errors,
        left,
        right,
        defect,
        new_left,
        new_right,
        d,
    )


    # ---------------------------------------------------------------------
    # Signal-bit noise
    #
    # Each stored signal bit is flipped independently after the
    # deterministic CA update.
    #
    # Signal noise is applied only to active trajectories.
    # ---------------------------------------------------------------------

    if threshold_sig > 0:

        for i in range(d):

            rng, noise_left = bernoulli64(
                rng,
                threshold_sig,
            )

            rng, noise_right = bernoulli64(
                rng,
                threshold_sig,
            )

            left[i] ^= (
                noise_left
                & active
            )

            right[i] ^= (
                noise_right
                & active
            )


    # ---------------------------------------------------------------------
    # Periodic signal reset
    #
    # On reset rounds, signal noise generated above is also cleared.
    # ---------------------------------------------------------------------

    if reset_now:

        for i in range(d):

            left[i] = np.uint64(0)
            right[i] = np.uint64(0)

    return rng


# =============================================================================
# Fast fixed-window first-passage kernel
# =============================================================================

@njit(parallel=True)
def hazard_kernel(
    d,
    threshold_data,
    threshold_meas,
    threshold_sig,
    shots,
    burn_rounds,
    reset_period,
    seed,
):
    """
    Fixed protocol:

        burn : [0, B)
        W1   : [B, 2B)
        W2   : [2B, 3B)

    Returns:

        failures[2]
        exposure[2]
    """

    batches = (
        shots + 63
    ) // 64

    nbits = max(
        1,
        int(
            math.ceil(
                math.log2(d + 1)
            )
        ),
    )

    failures_batch = np.zeros(
        (batches, 2),
        dtype=np.int64,
    )

    exposure_batch = np.zeros(
        (batches, 2),
        dtype=np.int64,
    )

    total_rounds = (
        3 * burn_rounds
    )

    for batch in prange(batches):

        first = (
            64 * batch
        )

        nlanes = min(
            64,
            shots - first,
        )

        if nlanes == 64:

            valid = ALL

        else:

            valid = (
                (
                    np.uint64(1)
                    << np.uint64(nlanes)
                )
                - np.uint64(1)
            )

        active = valid

        rng = (
            np.uint64(seed)
            + np.uint64(batch + 1)
            * np.uint64(
                0xD1B54A32D192ED03
            )
        )

        errors = np.zeros(
            d,
            dtype=np.uint64,
        )

        left = np.zeros(
            d,
            dtype=np.uint64,
        )

        right = np.zeros(
            d,
            dtype=np.uint64,
        )

        defect = np.empty(
            d,
            dtype=np.uint64,
        )

        new_left = np.empty(
            d,
            dtype=np.uint64,
        )

        new_right = np.empty(
            d,
            dtype=np.uint64,
        )

        counter = np.zeros(
            nbits,
            dtype=np.uint64,
        )

        for t in range(total_rounds):

            if active == 0:
                break

            round_number = (
                t + 1
            )

            # Window index:
            # -1 = burn-in
            #  0 = W1
            #  1 = W2

            if round_number <= burn_rounds:

                window = -1

            elif round_number <= 2 * burn_rounds:

                window = 0

            else:

                window = 1


            # -------------------------------------------------------------
            # Exposure at start of round
            # -------------------------------------------------------------

            if window >= 0:

                exposure_batch[
                    batch,
                    window,
                ] += popcount64(
                    active
                )


            reset_now = (
                round_number
                % reset_period
                == 0
            )


            # -------------------------------------------------------------
            # One noisy CA round
            # -------------------------------------------------------------

            rng = apply_round(
                errors=errors,
                left=left,
                right=right,
                defect=defect,
                new_left=new_left,
                new_right=new_right,
                d=d,
                active=active,
                rng=rng,
                threshold_data=(
                    threshold_data
                ),
                threshold_meas=(
                    threshold_meas
                ),
                threshold_sig=(
                    threshold_sig
                ),
                reset_now=reset_now,
            )


            # -------------------------------------------------------------
            # Absorbing logical-failure test
            # -------------------------------------------------------------

            failed = (
                majority_mask(
                    errors,
                    d,
                    active,
                    counter,
                )
                & active
            )

            if failed != 0:

                if window >= 0:

                    failures_batch[
                        batch,
                        window,
                    ] += popcount64(
                        failed
                    )

                active &= ~failed
                active &= valid


    failures = np.zeros(
        2,
        dtype=np.int64,
    )

    exposure = np.zeros(
        2,
        dtype=np.int64,
    )

    for batch in range(batches):

        failures[0] += (
            failures_batch[
                batch,
                0,
            ]
        )

        failures[1] += (
            failures_batch[
                batch,
                1,
            ]
        )

        exposure[0] += (
            exposure_batch[
                batch,
                0,
            ]
        )

        exposure[1] += (
            exposure_batch[
                batch,
                1,
            ]
        )

    return (
        failures,
        exposure,
    )


# =============================================================================
# Statistics
# =============================================================================

def hazard_stats(
    failures,
    exposure,
):

    failures = int(
        failures
    )

    exposure = int(
        exposure
    )

    if exposure <= 0:

        return (
            math.nan,
            math.nan,
        )

    h = (
        failures
        / exposure
    )

    if failures > 0:

        se = (
            math.sqrt(
                failures
            )
            / exposure
        )

    else:

        se = math.nan

    return (
        h,
        se,
    )


def point_statistics(
    failures,
    exposure,
    min_window_failures,
    z_max,
    rel_max,
):

    F1 = int(
        failures[0]
    )

    F2 = int(
        failures[1]
    )

    E1 = int(
        exposure[0]
    )

    E2 = int(
        exposure[1]
    )

    h1, se1 = hazard_stats(
        F1,
        E1,
    )

    h2, se2 = hazard_stats(
        F2,
        E2,
    )

    z = math.nan
    rel = math.nan

    if (
        np.isfinite(se1)
        and np.isfinite(se2)
    ):

        denom = math.sqrt(
            se1**2
            + se2**2
        )

        if denom > 0:

            z = (
                abs(h1 - h2)
                / denom
            )

    if (
        np.isfinite(h1)
        and np.isfinite(h2)
        and h1 + h2 > 0
    ):

        rel = (
            abs(h1 - h2)
            / (
                0.5
                * (
                    h1 + h2
                )
            )
        )

    enough_stats = (
        F1 >= min_window_failures
        and F2 >= min_window_failures
    )

    stationary = (
        enough_stats
        and np.isfinite(z)
        and np.isfinite(rel)
        and z <= z_max
        and rel <= rel_max
    )

    pooled_F = (
        F1 + F2
    )

    pooled_E = (
        E1 + E2
    )

    if (
        stationary
        and pooled_E > 0
    ):

        pL = (
            pooled_F
            / pooled_E
        )

        pL_se = (
            math.sqrt(
                pooled_F
            )
            / pooled_E
        )

    else:

        pL = math.nan
        pL_se = math.nan

    return {
        "F1": F1,
        "E1": E1,
        "h1": h1,
        "se1": se1,

        "F2": F2,
        "E2": E2,
        "h2": h2,
        "se2": se2,

        "z": z,
        "rel": rel,

        "stationary": stationary,

        "pL": pL,
        "pL_se": pL_se,
    }


# =============================================================================
# CSV
# =============================================================================

COLUMNS = [
    "d",
    "p",
    "q",
    "p_sig",

    "reset_factor",
    "reset_period",

    "burn_reset",
    "burn_rounds",

    "shots",

    "F1",
    "E1",
    "h1",
    "se1",

    "F2",
    "E2",
    "h2",
    "se2",

    "z_drift",
    "rel_drift",

    "stationary",

    "pL",
    "pL_se",
]


KEY = [
    "d",
    "p",
    "q",
    "p_sig",
    "reset_factor",
    "reset_period",
    "burn_reset",
]


def read_output(path):

    path = Path(
        path
    )

    if not path.exists():

        return pd.DataFrame(
            columns=COLUMNS
        )

    try:

        df = pd.read_csv(
            path
        )

        # -------------------------------------------------------------
        # Backward compatibility:
        # older CSV files without p_sig are interpreted as p_sig=0.
        # -------------------------------------------------------------

        if "p_sig" not in df.columns:
            df["p_sig"] = 0.0

    except pd.errors.EmptyDataError:

        return pd.DataFrame(
            columns=COLUMNS
        )

    return df


def merge_result(
    path,
    row,
):

    old = read_output(
        path
    )

    if old.empty:

        clean_row = {
            k: v
            for k, v in row.items()
            if not k.startswith("_")
        }

        out = pd.DataFrame(
            [clean_row],
            columns=COLUMNS,
        )

        atomic_write_csv(
            out,
            path,
        )

        return


    mask = (
        (
            old["d"].astype(int)
            == int(row["d"])
        )
        & np.isclose(
            old["p"],
            row["p"],
            atol=1e-12,
            rtol=0,
        )
        & np.isclose(
            old["q"],
            row["q"],
            atol=1e-12,
            rtol=0,
        )
        & np.isclose(
            old["p_sig"],
            row["p_sig"],
            atol=1e-12,
            rtol=0,
        )
        & np.isclose(
            old["reset_factor"],
            row["reset_factor"],
            atol=1e-12,
            rtol=0,
        )
        & (
            old["reset_period"].astype(int)
            == int(
                row["reset_period"]
            )
        )
        & (
            old["burn_reset"].astype(int)
            == int(
                row["burn_reset"]
            )
        )
    )


    if np.any(mask):

        idx = old.index[
            mask
        ][0]


        old_shots = int(
            old.loc[
                idx,
                "shots",
            ]
        )

        old_F1 = int(
            old.loc[
                idx,
                "F1",
            ]
        )

        old_E1 = int(
            old.loc[
                idx,
                "E1",
            ]
        )

        old_F2 = int(
            old.loc[
                idx,
                "F2",
            ]
        )

        old_E2 = int(
            old.loc[
                idx,
                "E2",
            ]
        )


        row["shots"] += (
            old_shots
        )

        row["F1"] += (
            old_F1
        )

        row["E1"] += (
            old_E1
        )

        row["F2"] += (
            old_F2
        )

        row["E2"] += (
            old_E2
        )


        # -------------------------------------------------------------
        # Recompute all statistics from pooled raw counts.
        # -------------------------------------------------------------

        stats = point_statistics(
            failures=np.array(
                [
                    row["F1"],
                    row["F2"],
                ],
                dtype=np.int64,
            ),
            exposure=np.array(
                [
                    row["E1"],
                    row["E2"],
                ],
                dtype=np.int64,
            ),
            min_window_failures=(
                row[
                    "_min_window_failures"
                ]
            ),
            z_max=(
                row["_z_max"]
            ),
            rel_max=(
                row["_rel_max"]
            ),
        )


        row["h1"] = (
            stats["h1"]
        )

        row["se1"] = (
            stats["se1"]
        )

        row["h2"] = (
            stats["h2"]
        )

        row["se2"] = (
            stats["se2"]
        )

        row["z_drift"] = (
            stats["z"]
        )

        row["rel_drift"] = (
            stats["rel"]
        )

        row["stationary"] = (
            stats["stationary"]
        )

        row["pL"] = (
            stats["pL"]
        )

        row["pL_se"] = (
            stats["pL_se"]
        )


        old = old.loc[
            ~mask
        ].copy()


    clean_row = {
        k: v
        for k, v in row.items()
        if not k.startswith("_")
    }


    out = pd.concat(
        [
            old,
            pd.DataFrame(
                [clean_row],
                columns=COLUMNS,
            ),
        ],
        ignore_index=True,
    )


    out = (
        out
        .sort_values(
            [
                "d",
                "p",
                "q",
                "p_sig",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    atomic_write_csv(
        out,
        path,
    )


# =============================================================================
# One point
# =============================================================================

def run_point(
    d,
    p,
    q,
    p_sig,
    reset_factor,
    burn_reset,
    shots,
    min_window_failures,
    z_max,
    rel_max,
    seed,
    output,
):

    reset_period = max(
        1,
        int(
            round(
                reset_factor
                * d
            )
        ),
    )

    burn_rounds = (
        burn_reset
        * reset_period
    )


    td = probability_threshold(
        p
    )

    tq = probability_threshold(
        q
    )

    ts = probability_threshold(
        p_sig
    )


    batch_seed = deterministic_seed(
        seed,
        d,
        f"{p:.12g}",
        f"{q:.12g}",
        f"{p_sig:.12g}",
        reset_period,
        burn_reset,
        shots,
    )


    failures, exposure = hazard_kernel(
        d=d,
        threshold_data=td,
        threshold_meas=tq,
        threshold_sig=ts,
        shots=shots,
        burn_rounds=burn_rounds,
        reset_period=reset_period,
        seed=batch_seed,
    )


    stats = point_statistics(
        failures=failures,
        exposure=exposure,
        min_window_failures=(
            min_window_failures
        ),
        z_max=z_max,
        rel_max=rel_max,
    )


    status = (
        "OK"
        if stats["stationary"]
        else "UNRESOLVED"
    )


    print(
        f"    W1: "
        f"F={stats['F1']:7d}  "
        f"E={stats['E1']:12d}  "
        f"h={stats['h1']:.6e}"
    )

    print(
        f"    W2: "
        f"F={stats['F2']:7d}  "
        f"E={stats['E2']:12d}  "
        f"h={stats['h2']:.6e}"
    )

    print(
        f"    drift: "
        f"z={stats['z']:.2f}  "
        f"rel={stats['rel']:.3f}  "
        f"{status}"
    )


    if stats["stationary"]:

        print(
            f"    pL={stats['pL']:.6e} "
            f"+- {stats['pL_se']:.2e}"
        )


    row = {
        "d": int(d),

        "p": float(p),

        "q": float(q),

        "p_sig": float(
            p_sig
        ),

        "reset_factor":
            float(
                reset_factor
            ),

        "reset_period":
            int(
                reset_period
            ),

        "burn_reset":
            int(
                burn_reset
            ),

        "burn_rounds":
            int(
                burn_rounds
            ),

        "shots":
            int(
                shots
            ),

        "F1":
            int(
                stats["F1"]
            ),

        "E1":
            int(
                stats["E1"]
            ),

        "h1":
            float(
                stats["h1"]
            ),

        "se1":
            float(
                stats["se1"]
            ),

        "F2":
            int(
                stats["F2"]
            ),

        "E2":
            int(
                stats["E2"]
            ),

        "h2":
            float(
                stats["h2"]
            ),

        "se2":
            float(
                stats["se2"]
            ),

        "z_drift":
            float(
                stats["z"]
            ),

        "rel_drift":
            float(
                stats["rel"]
            ),

        "stationary":
            bool(
                stats["stationary"]
            ),

        "pL":
            float(
                stats["pL"]
            ),

        "pL_se":
            float(
                stats["pL_se"]
            ),

        "_min_window_failures":
            int(
                min_window_failures
            ),

        "_z_max":
            float(
                z_max
            ),

        "_rel_max":
            float(
                rel_max
            ),
    }


    merge_result(
        output,
        row,
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--d",
        nargs="+",
        type=int,
        required=True,
    )


    parser.add_argument(
        "--p",
        nargs="+",
        type=float,
        required=True,
    )


    parser.add_argument(
        "--q",
        nargs="+",
        type=float,
        default=None,
        help="Default: q=p.",
    )


    parser.add_argument(
        "--p-sig",
        nargs="+",
        type=float,
        default=None,
        help="Default: p_sig=p.",
    )


    parser.add_argument(
        "--reset-factor",
        type=float,
        default=0.35,
    )


    parser.add_argument(
        "--burn-reset",
        type=int,
        default=10,
        help=(
            "Fixed burn-in/window length "
            "in reset periods."
        ),
    )


    parser.add_argument(
        "--shots",
        type=int,
        default=100000,
    )


    parser.add_argument(
        "--min-window-failures",
        type=int,
        default=100,
    )


    parser.add_argument(
        "--z-max",
        type=float,
        default=2.0,
    )


    parser.add_argument(
        "--rel-max",
        type=float,
        default=0.15,
    )


    parser.add_argument(
        "--threads",
        type=int,
        default=0,
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=BASE_SEED,
    )


    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )


    return parser.parse_args()


def main():

    args = parse_args()


    if args.threads > 0:

        set_num_threads(
            args.threads
        )


    print()
    print("=" * 100)

    print(
        "SCALA1D PHENOMENOLOGICAL + SIGNAL-NOISE "
        "FIXED-WINDOW FIRST-PASSAGE HAZARD"
    )

    print("=" * 100)


    print(
        f"burn/window length = "
        f"{args.burn_reset} t_R"
    )

    print(
        f"shots / point      = "
        f"{args.shots}"
    )

    print(
        f"output             = "
        f"{args.output}"
    )


    for d in args.d:

        if (
            d <= 0
            or d % 2 == 0
        ):

            raise ValueError(
                f"d must be positive odd; got {d}"
            )


        for p in args.p:


            if args.q is None:

                q_values = [
                    p
                ]

            else:

                q_values = (
                    args.q
                )


            if args.p_sig is None:

                p_sig_values = [
                    p
                ]

            else:

                p_sig_values = (
                    args.p_sig
                )


            for q in q_values:

                for p_sig in p_sig_values:


                    reset_period = max(
                        1,
                        int(
                            round(
                                args.reset_factor
                                * d
                            )
                        ),
                    )


                    B = (
                        args.burn_reset
                        * reset_period
                    )


                    print()

                    print(
                        f"d={d:3d}  "
                        f"p={p:.6f}  "
                        f"q={q:.6f}  "
                        f"p_sig={p_sig:.6f}  "
                        f"tR={reset_period:3d}  "
                        f"B={B:5d}  "
                        f"T={3*B:5d}"
                    )


                    run_point(
                        d=d,

                        p=float(
                            p
                        ),

                        q=float(
                            q
                        ),

                        p_sig=float(
                            p_sig
                        ),

                        reset_factor=(
                            args.reset_factor
                        ),

                        burn_reset=(
                            args.burn_reset
                        ),

                        shots=(
                            args.shots
                        ),

                        min_window_failures=(
                            args.min_window_failures
                        ),

                        z_max=(
                            args.z_max
                        ),

                        rel_max=(
                            args.rel_max
                        ),

                        seed=(
                            args.seed
                        ),

                        output=(
                            args.output
                        ),
                    )


    print()
    print("=" * 100)

    print(
        f"Saved: {args.output}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
