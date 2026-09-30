"""
Harrington1D phenomenological-noise hazard runner.

Simulation only. No plotting.

Default scan
------------
Left panel:
    p = q = 0.015
    d = 3, 9, 27, 81
    t_max = 2000

Right panel:
    d = 27
    p = q = 0.015, 0.020, 0.025, 0.030
    t_max = 1000

Output
------
One row per time bin:

    d, p, q, rounds,
    t_start, t_end,
    shots, at_risk, failures,
    survivors_end, exposure

The logical hazard is reconstructed later as

    h = failures / exposure.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from src.harrington1d_optimized import (
    MASK64,
    _bernoulli_word,
    _build_geometry,
    _seed_rng,
    _step,
)


# =============================================================================
# Utilities
# =============================================================================

def probability_threshold(p: float) -> int:
    if p <= 0.0:
        return 0

    if p >= 1.0:
        return 0x100000000

    return int(
        math.floor(
            p * 4294967296.0
        )
    )


@njit(inline="always")
def logical_mask(
    qubits,
    d,
    valid_mask,
):
    """
    Logical failure for odd repetition-code distance d:

        weight(error) > d / 2.
    """

    out = np.uint64(0)

    for lane in range(64):

        bit = (
            np.uint64(1)
            << np.uint64(lane)
        )

        if (valid_mask & bit) == 0:
            continue

        weight = 0

        for i in range(d):
            if qubits[i] & bit:
                weight += 1

        if weight > d // 2:
            out |= bit

    return out


# =============================================================================
# Packed simulation
# =============================================================================

@njit
def simulate_group(
    d,
    rounds,
    p_threshold,
    q_threshold,
    valid_mask,
    rng_state,
    n_levels,
    site_to_rep,
    roles,
    U_levels,
    Q_levels,
    periods,
    nbits_levels,
    threshold_N,
    threshold_C,
    rep_count,
    max_bits,
):
    qubits = np.zeros(
        d,
        dtype=np.uint64,
    )

    defects = np.zeros(
        d,
        dtype=np.uint64,
    )

    count_L = np.zeros(
        (n_levels, d),
        dtype=np.uint64,
    )
    count_R = np.zeros_like(count_L)

    flip_L = np.zeros_like(count_L)
    flip_R = np.zeros_like(count_L)

    new_count_L = np.zeros_like(count_L)
    new_count_R = np.zeros_like(count_L)

    new_flip_L = np.zeros_like(count_L)
    new_flip_R = np.zeros_like(count_L)

    counters = np.zeros(
        (
            3,
            max_bits,
            rep_count,
        ),
        dtype=np.uint64,
    )

    ages = np.zeros(
        n_levels,
        dtype=np.int64,
    )

    first_failure = np.zeros(
        64,
        dtype=np.int64,
    )

    alive = valid_mask

    for t in range(
        1,
        rounds + 1,
    ):

        if alive == 0:
            break

        # ---------------------------------------------------------------------
        # Data noise
        # ---------------------------------------------------------------------

        if p_threshold != 0:

            for i in range(d):

                qubits[i] ^= (
                    _bernoulli_word(
                        rng_state,
                        p_threshold,
                    )
                    & alive
                )

        # ---------------------------------------------------------------------
        # Syndrome + measurement noise
        # ---------------------------------------------------------------------

        for i in range(d):

            right = (
                i + 1
                if i + 1 < d
                else 0
            )

            defects[i] = (
                qubits[i]
                ^ qubits[right]
            )

            if q_threshold != 0:

                defects[i] ^= (
                    _bernoulli_word(
                        rng_state,
                        q_threshold,
                    )
                    & alive
                )

        # ---------------------------------------------------------------------
        # Harrington CA update
        # ---------------------------------------------------------------------

        _step(
            qubits,
            defects,

            count_L,
            count_R,
            flip_L,
            flip_R,

            new_count_L,
            new_count_R,
            new_flip_L,
            new_flip_R,

            counters,

            site_to_rep,
            roles,
            ages,
            U_levels,
            Q_levels,
            periods,
            nbits_levels,
            threshold_N,
            threshold_C,

            alive,

            0,              # no internal signal noise
            rng_state,

            d,
            n_levels,

            True,           # syndrome already computed
        )

        # ---------------------------------------------------------------------
        # First logical failure
        # ---------------------------------------------------------------------

        failed_now = (
            logical_mask(
                qubits,
                d,
                alive,
            )
            & alive
        )

        if failed_now == 0:
            continue

        for lane in range(64):

            bit = (
                np.uint64(1)
                << np.uint64(lane)
            )

            if failed_now & bit:
                first_failure[lane] = t

        alive &= ~failed_now

    return first_failure


def simulate(
    d: int,
    p: float,
    q: float,
    shots: int,
    rounds: int,
    seed: int,
    U: int,
    fN: float,
    fC: float,
):
    geometry = _build_geometry(
        d,
        U,
        fN,
        fC,
    )

    rng_state = _seed_rng(
        seed
    )

    p_threshold = probability_threshold(
        p
    )

    q_threshold = probability_threshold(
        q
    )

    failure_times = np.zeros(
        shots,
        dtype=np.int64,
    )

    offset = 0

    while offset < shots:

        n = min(
            64,
            shots - offset,
        )

        valid_mask = (
            MASK64
            if n == 64
            else np.uint64(
                (1 << n) - 1
            )
        )

        result = simulate_group(
            d,
            rounds,
            p_threshold,
            q_threshold,
            valid_mask,
            rng_state,

            geometry["n_levels"],
            geometry["site_to_rep"],
            geometry["roles"],
            geometry["U_levels"],
            geometry["Q_levels"],
            geometry["periods"],
            geometry["nbits_levels"],
            geometry["threshold_N"],
            geometry["threshold_C"],
            geometry["rep_count"],
            geometry["max_bits"],
        )

        failure_times[
            offset:
            offset + n
        ] = result[:n]

        offset += n

    return failure_times


# =============================================================================
# Time-bin sufficient statistics
# =============================================================================

def summarize_failure_times(
    failure_times,
    d,
    p,
    q,
    rounds,
    bin_width,
):
    rows = []

    shots = len(
        failure_times
    )

    for t0 in range(
        0,
        rounds,
        bin_width,
    ):

        t1 = min(
            t0 + bin_width,
            rounds,
        )

        alive = (
            (failure_times == 0)
            | (failure_times > t0)
        )

        at_risk = int(
            np.count_nonzero(
                alive
            )
        )

        if at_risk == 0:
            break

        failed = (
            (failure_times > t0)
            & (failure_times <= t1)
        )

        failures = int(
            np.count_nonzero(
                failed
            )
        )

        surviving_end = (
            (failure_times == 0)
            | (failure_times > t1)
        )

        survivors_end = int(
            np.count_nonzero(
                surviving_end
            )
        )

        times = failure_times[
            alive
        ]

        exposure = int(
            np.sum(
                np.where(
                    (times == 0)
                    | (times > t1),
                    t1 - t0,
                    times - t0,
                ),
                dtype=np.int64,
            )
        )

        rows.append(
            {
                "d": int(d),
                "p": float(p),
                "q": float(q),
                "rounds": int(rounds),

                "t_start": int(t0),
                "t_end": int(t1),

                "shots": int(shots),
                "at_risk": int(at_risk),
                "failures": int(failures),
                "survivors_end": int(
                    survivors_end
                ),
                "exposure": int(exposure),
            }
        )

    return rows


# =============================================================================
# Scan
# =============================================================================

def build_scan(
    left_distances,
    left_p,
    left_rounds,
    right_d,
    right_p_values,
    right_rounds,
):
    """
    Deduplicate equal (d,p,q) configurations.

    If the same point occurs in both panels, simulate it only once,
    using the larger requested runtime.
    """

    configs = {}

    for d in left_distances:

        key = (
            int(d),
            float(left_p),
            float(left_p),
        )

        configs[key] = max(
            configs.get(
                key,
                0,
            ),
            int(left_rounds),
        )

    for p in right_p_values:

        key = (
            int(right_d),
            float(p),
            float(p),
        )

        configs[key] = max(
            configs.get(
                key,
                0,
            ),
            int(right_rounds),
        )

    return [
        {
            "d": d,
            "p": p,
            "q": q,
            "rounds": rounds,
        }
        for (
            d,
            p,
            q,
        ), rounds in sorted(
            configs.items()
        )
    ]


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Harrington1D phenomenological "
            "hazard scan."
        )
    )

    parser.add_argument(
        "--shots",
        type=int,
        default=1_000_000,
    )

    parser.add_argument(
        "--bin-width",
        type=int,
        default=50,
    )

    # -------------------------------------------------------------------------
    # Left panel
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--left-distances",
        nargs="+",
        type=int,
        default=[
            3,
            9,
            27,
            81,
        ],
    )

    parser.add_argument(
        "--left-p",
        type=float,
        default=0.015,
    )

    parser.add_argument(
        "--left-rounds",
        type=int,
        default=2000,
    )

    # -------------------------------------------------------------------------
    # Right panel
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--right-d",
        type=int,
        default=27,
    )

    parser.add_argument(
        "--right-p-values",
        nargs="+",
        type=float,
        default=[
            0.015,
            0.020,
            0.025,
            0.030,
        ],
    )

    parser.add_argument(
        "--right-rounds",
        type=int,
        default=1000,
    )

    # -------------------------------------------------------------------------
    # Harrington parameters
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--U",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--fN",
        type=float,
        default=0.4,
    )

    parser.add_argument(
        "--fC",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/pheno/"
            "pheno_hazard_scan.csv"
        ),
    )

    parser.add_argument(
        "--append",
        action="store_true",
        help=(
            "Append this independent simulation batch "
            "to an existing CSV instead of overwriting it."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    configs = build_scan(
        left_distances=(
            args.left_distances
        ),
        left_p=args.left_p,
        left_rounds=args.left_rounds,
        right_d=args.right_d,
        right_p_values=(
            args.right_p_values
        ),
        right_rounds=args.right_rounds,
    )

    print()
    print("=" * 100)
    print(
        "HARRINGTON1D PHENOMENOLOGICAL HAZARD SCAN"
    )
    print("=" * 100)

    print(
        f"shots / point      = {args.shots}"
    )

    print(
        f"bin width          = {args.bin_width}"
    )

    print(
        f"U                  = {args.U}"
    )

    print(
        f"fN                 = {args.fN}"
    )

    print(
        f"fC                 = {args.fC}"
    )

    print(
        f"output             = {args.output}"
    )

    print(
        f"append             = {args.append}"
    )

    print(
        f"parameter points   = {len(configs)}"
    )

    print("-" * 100)

    all_rows = []

    for j, config in enumerate(
        configs,
        start=1,
    ):
        d = config["d"]
        p = config["p"]
        q = config["q"]
        rounds = config["rounds"]

        seed = (
            args.seed
            + 10007 * j
            + 101 * d
            + int(
                round(
                    1e6 * p
                )
            )
        )

        print(
            f"[{j:2d}/{len(configs):2d}] "
            f"d={d:3d}  "
            f"p=q={p:.5f}  "
            f"T={rounds:5d}  "
            f"N={args.shots:8d}"
        )

        failure_times = simulate(
            d=d,
            p=p,
            q=q,
            shots=args.shots,
            rounds=rounds,
            seed=seed,
            U=args.U,
            fN=args.fN,
            fC=args.fC,
        )

        rows = summarize_failure_times(
            failure_times=failure_times,
            d=d,
            p=p,
            q=q,
            rounds=rounds,
            bin_width=args.bin_width,
        )

        all_rows.extend(
            rows
        )

        failed = int(
            np.count_nonzero(
                failure_times
            )
        )

        censored = (
            args.shots
            - failed
        )

        print(
            f"        failed={failed:8d}  "
            f"censored={censored:8d}  "
            f"failure fraction="
            f"{failed / args.shots:.5f}"
        )

    new_df = pd.DataFrame(
        all_rows
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        args.append
        and args.output.exists()
    ):
        old_df = pd.read_csv(
            args.output
        )

        out_df = pd.concat(
            [
                old_df,
                new_df,
            ],
            ignore_index=True,
        )

    else:
        out_df = new_df

    out_df.to_csv(
        args.output,
        index=False,
    )

    print("-" * 100)

    print(
        f"Saved: {args.output}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
