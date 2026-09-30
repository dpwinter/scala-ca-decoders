from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_pheno_hazard import (
    actual_logicals,
    advance_group,
    build_matching,
    make_group,
    unpack_syndrome,
)


# =============================================================================
# Exact discrete SCALA witness weight from Eq. (2)
# =============================================================================

def factor_pairs(w: int):
    """
    Return all ordered factor pairs (k, ne) with k * ne = w.
    """
    out = []

    for k in range(1, w + 1):
        if w % k == 0:
            ne = w // k
            out.append((k, ne))

    return out


def f_integer(w: int) -> int:
    """
    f(w) = min_{k ne = w} (k + ne)
    """
    return min(
        k + ne
        for k, ne in factor_pairs(w)
    )


def witness_weight(d: int) -> int:
    """
    Eq. (2):

        w0(d) = min {
            w in Z_+ :
            2 w - f(w) + 1 >= (d+1)/2
        }.
    """

    target = (d + 1) // 2

    w = 1

    while True:

        wmax = (
            2 * w
            - f_integer(w)
            + 1
        )

        if wmax >= target:
            return w

        w += 1


def optimal_factor_pairs(w: int):
    """
    Factor pairs attaining f(w).

    We test all of them, because (k,ne) and (ne,k)
    can represent dynamically distinct cluster patterns.
    """

    fmin = f_integer(w)

    return [
        (k, ne)
        for k, ne in factor_pairs(w)
        if k + ne == fmin
    ]


# =============================================================================
# Witness construction
# =============================================================================

def witness_positions(
    d: int,
    k: int,
    ne: int,
    offset: int = 0,
):
    """
    k clusters, each containing ne consecutive errors,
    separated by ne-1 clean qubits.

    Returns 1D edge coordinates modulo d.
    """

    positions = []

    x = offset

    for cluster in range(k):

        for j in range(ne):
            positions.append(
                (x + j) % d
            )

        x += (
            ne
            + (ne - 1)
        )

    # The construction should not overlap itself for the
    # distances of interest. Catch accidental wrap overlap.
    if len(set(positions)) != len(positions):
        raise RuntimeError(
            f"Witness overlaps itself for "
            f"d={d}, k={k}, ne={ne}: "
            f"{positions}"
        )

    return positions


def inject_witness(
    state,
    d: int,
    k: int,
    ne: int,
    orientation: str,
    offset: int = 0,
):
    """
    Embed the SCALA1D witness into one nontrivial 1D
    cross-section of the toric code.

    Horizontal:
        errors on h[0,x]

    Vertical:
        errors on v[y,0]
    """

    pos = witness_positions(
        d=d,
        k=k,
        ne=ne,
        offset=offset,
    )

    bit = np.uint64(1)

    if orientation == "horizontal":

        for x in pos:
            state["h"][0, x] ^= bit

    elif orientation == "vertical":

        for y in pos:
            state["v"][y, 0] ^= bit

    else:
        raise ValueError(
            f"Unknown orientation: {orientation}"
        )

    return pos


# =============================================================================
# Logical checker
# =============================================================================

def logical_failure(
    state,
    matching,
):
    """
    Same instantaneous logical-error definition used by
    run_pheno_hazard.py:

        actual logical class XOR MWPM-predicted class.
    """

    lanes = np.asarray(
        [0],
        dtype=np.int64,
    )

    syndrome = unpack_syndrome(
        state["true_defect"],
        lanes,
    )

    actual = actual_logicals(
        state["h"],
        state["v"],
        lanes,
    )

    predicted = np.asarray(
        matching.decode_batch(
            syndrome
        ),
        dtype=np.uint8,
    )

    if predicted.ndim == 1:
        predicted = predicted[:, None]

    failed = np.any(
        predicted != actual,
        axis=1,
    )

    return bool(
        failed[0]
    )


# =============================================================================
# One deterministic witness test
# =============================================================================

def run_one(
    d: int,
    reset_period: int,
    reset_phase: int,
    k: int,
    ne: int,
    orientation: str,
    max_rounds: int,
):
    """
    Start with one exact low-weight data-error witness,
    clean signals and no further physical/measurement noise.

    reset_phase:
        number of elapsed rounds since the previous reset
        at the moment the witness is injected.

    Thus all possible positions of the witness relative to
    the periodic reset schedule are tested.
    """

    state = make_group(
        d=d,
        n_lanes=1,
        seed=1,
    )

    # Inject witness into otherwise clean data.
    positions = inject_witness(
        state=state,
        d=d,
        k=k,
        ne=ne,
        orientation=orientation,
        offset=0,
    )

    # Set phase within reset cycle.
    state["reset_clock"] = int(
        reset_phase
    )

    # Initial exact syndrome.
    # advance_group() recomputes it after the first update,
    # so we only need the checker after each CA step.
    matching = build_matching(
        d
    )

    failure_time = None

    # No further noise.
    p_threshold = 0
    q_threshold = 0

    for t in range(
        1,
        max_rounds + 1,
    ):

        advance_group(
            state=state,
            d=d,
            p_threshold=p_threshold,
            q_threshold=q_threshold,
            reset_period=reset_period,
        )

        if logical_failure(
            state=state,
            matching=matching,
        ):

            failure_time = t
            break

    return {
        "failed":
            failure_time is not None,

        "failure_time":
            (
                failure_time
                if failure_time is not None
                else np.nan
            ),

        "positions":
            positions,
    }


# =============================================================================
# Scan
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--d",
        nargs="+",
        type=int,
        default=[
            21,
            41,
            61,
        ],
    )

    parser.add_argument(
        "--alpha",
        nargs="+",
        type=float,
        default=[
            0.20,
            0.25,
            0.30,
            0.35,
        ],
    )

    parser.add_argument(
        "--rounds-multiplier",
        type=float,
        default=2.0,
        help=(
            "Maximum noiseless evolution time "
            "in units of d."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/pheno/"
            "scala2d_tR_witness_test.csv"
        ),
    )

    args = parser.parse_args()

    rows = []

    print()
    print("=" * 100)
    print(
        "SCALA2D FIXED-RESET MINIMUM-WEIGHT WITNESS TEST"
    )
    print("=" * 100)

    for d in args.d:

        w0 = witness_weight(
            d
        )

        pairs = optimal_factor_pairs(
            w0
        )

        target = (
            d + 1
        ) // 2

        print()
        print(
            f"d={d}"
        )
        print(
            f"    predicted w0       = {w0}"
        )
        print(
            f"    target logical wt  = {target}"
        )
        print(
            f"    optimal factors    = {pairs}"
        )

        max_rounds = int(
            math.ceil(
                args.rounds_multiplier
                * d
            )
        )

        for alpha in args.alpha:

            tR = max(
                1,
                int(
                    round(
                        alpha * d
                    )
                ),
            )

            n_tests = 0
            n_fail = 0

            earliest = None

            print()
            print(
                f"    alpha={alpha:.3f} "
                f"-> tR={tR}"
            )

            for (
                k,
                ne,
            ) in pairs:

                for orientation in [
                    "horizontal",
                    "vertical",
                ]:

                    # ---------------------------------------------
                    # Scan every possible position in reset cycle.
                    # ---------------------------------------------

                    for phase in range(
                        tR
                    ):

                        result = run_one(
                            d=d,
                            reset_period=tR,
                            reset_phase=phase,
                            k=k,
                            ne=ne,
                            orientation=orientation,
                            max_rounds=max_rounds,
                        )

                        n_tests += 1

                        if result["failed"]:

                            n_fail += 1

                            ft = int(
                                result[
                                    "failure_time"
                                ]
                            )

                            if (
                                earliest is None
                                or ft < earliest
                            ):
                                earliest = ft

                        rows.append(
                            {
                                "d":
                                    int(d),

                                "w0":
                                    int(w0),

                                "k":
                                    int(k),

                                "ne":
                                    int(ne),

                                "alpha_requested":
                                    float(alpha),

                                "reset_period":
                                    int(tR),

                                "actual_alpha":
                                    float(
                                        tR / d
                                    ),

                                "orientation":
                                    orientation,

                                "reset_phase":
                                    int(phase),

                                "failed":
                                    bool(
                                        result[
                                            "failed"
                                        ]
                                    ),

                                "failure_time":
                                    result[
                                        "failure_time"
                                    ],

                                "max_rounds":
                                    int(
                                        max_rounds
                                    ),
                            }
                        )

            frac = (
                n_fail
                / n_tests
            )

            print(
                f"        failing phases = "
                f"{n_fail}/{n_tests} "
                f"({frac:.3f})"
            )

            if earliest is not None:

                print(
                    f"        earliest failure = "
                    f"t={earliest}"
                )

            else:

                print(
                    "        NO witness failure"
                )

    # -------------------------------------------------------------------------
    # Append/replace safely into one canonical CSV
    # -------------------------------------------------------------------------

    new = pd.DataFrame(
        rows
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    key = [
        "d",
        "w0",
        "k",
        "ne",
        "reset_period",
        "orientation",
        "reset_phase",
    ]

    if args.output.exists():

        old = pd.read_csv(
            args.output
        )

        new_keys = set(
            new[
                key
            ].itertuples(
                index=False,
                name=None,
            )
        )

        keep = [
            tuple(row)
            not in new_keys
            for row in old[
                key
            ].itertuples(
                index=False,
                name=None,
            )
        ]

        old = old[
            np.asarray(
                keep,
                dtype=bool,
            )
        ]

        out = pd.concat(
            [
                old,
                new,
            ],
            ignore_index=True,
        )

    else:

        out = new

    out.to_csv(
        args.output,
        index=False,
    )

    print()
    print("-" * 100)
    print(
        f"Saved: {args.output}"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
