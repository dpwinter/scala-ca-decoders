"""
Harrington2D phenomenological-noise hazard runner.

Logical failure definition
--------------------------
After every complete Harrington CA step:

1. take the current physical data-error configuration,
2. compute its exact/noise-free toric-code syndrome,
3. decode that syndrome with MWPM (PyMatching),
4. compare the logical class predicted by MWPM with the
   actual logical class of the data-error configuration,
5. record the first round where they disagree.

Thus measurement noise affects the CA through q, but the offline
logical checker always receives the exact current data-error syndrome.

Output
------
Time-binned first-passage sufficient statistics:

    d, p, q, rounds,
    t_start, t_end,
    shots, at_risk, failures,
    survivors_end, exposure

The plotting script reconstructs

    h(t) = failures / exposure.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pymatching
from numba import njit

from src.harrington2d_optimized import (
    MASK64,
    N_DIR,
    N_CARD,
    _bernoulli_word,
    _build_geometry,
    _seed_rng,
    _step,
    _syndrome,
)


# =============================================================================
# Probability -> uint32 threshold
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


# =============================================================================
# Toric-code MWPM checker
# =============================================================================

def node(y: int, x: int, d: int) -> int:
    return y * d + x


def build_matching(d: int) -> pymatching.Matching:
    """
    Build MWPM graph for the periodic d x d toric code.

    Syndrome convention from harrington2d_optimized.py:

        s[y,x]
          = h[y,x] ^ h[y,x+1]
            ^ v[y-1,x] ^ v[y,x].

    Hence

        h[y,x] touches s[y,x] and s[y,x-1],
        v[y,x] touches s[y,x] and s[y+1,x].

    fault_id 0:
        parity across horizontal[:, 0]

    fault_id 1:
        parity across vertical[0, :]
    """

    matching = pymatching.Matching()

    # Horizontal data qubits.
    for y in range(d):
        for x in range(d):

            xm = (
                x - 1
                if x > 0
                else d - 1
            )

            faults = (
                {0}
                if x == 0
                else set()
            )

            matching.add_edge(
                node(y, x, d),
                node(y, xm, d),
                fault_ids=faults,
                weight=1.0,
            )

    # Vertical data qubits.
    for y in range(d):
        yp = (
            y + 1
            if y + 1 < d
            else 0
        )

        for x in range(d):

            faults = (
                {1}
                if y == 0
                else set()
            )

            matching.add_edge(
                node(y, x, d),
                node(yp, x, d),
                fault_ids=faults,
                weight=1.0,
            )

    return matching


# =============================================================================
# Noise + measured syndrome
# =============================================================================

@njit
def apply_noise_and_measure(
    horizontal,
    vertical,
    true_defects,
    measured_defects,
    p_threshold,
    q_threshold,
    alive,
    rng_state,
    d,
):
    """
    Apply phenomenological data noise, compute the exact syndrome,
    then corrupt that syndrome for the Harrington decoder.
    """

    # -------------------------------------------------------------------------
    # Data noise
    # -------------------------------------------------------------------------

    if p_threshold != 0:

        for y in range(d):
            for x in range(d):

                horizontal[y, x] ^= (
                    _bernoulli_word(
                        rng_state,
                        p_threshold,
                    )
                    & alive
                )

                vertical[y, x] ^= (
                    _bernoulli_word(
                        rng_state,
                        p_threshold,
                    )
                    & alive
                )

    # -------------------------------------------------------------------------
    # Exact syndrome
    # -------------------------------------------------------------------------

    _syndrome(
        horizontal,
        vertical,
        true_defects,
        d,
    )

    # -------------------------------------------------------------------------
    # Measurement noise used by the CA
    # -------------------------------------------------------------------------

    for y in range(d):
        for x in range(d):

            measured_defects[y, x] = (
                true_defects[y, x]
            )

            if q_threshold != 0:

                measured_defects[y, x] ^= (
                    _bernoulli_word(
                        rng_state,
                        q_threshold,
                    )
                    & alive
                )


# =============================================================================
# One packed Harrington group
# =============================================================================

def make_group(
    d,
    n_lanes,
    geometry,
    seed,
):

    n_levels = geometry[
        "n_levels"
    ]

    rep_count = geometry[
        "rep_count"
    ]

    max_bits = geometry[
        "max_bits"
    ]

    valid_mask = (
        MASK64
        if n_lanes == 64
        else np.uint64(
            (1 << n_lanes) - 1
        )
    )

    count_sig = np.zeros(
        (
            n_levels,
            d,
            d,
            N_DIR,
        ),
        dtype=np.uint64,
    )

    flip_sig = np.zeros(
        (
            n_levels,
            d,
            d,
            N_CARD,
        ),
        dtype=np.uint64,
    )

    return {
        "horizontal":
            np.zeros(
                (d, d),
                dtype=np.uint64,
            ),

        "vertical":
            np.zeros(
                (d, d),
                dtype=np.uint64,
            ),

        "true_defects":
            np.zeros(
                (d, d),
                dtype=np.uint64,
            ),

        "measured_defects":
            np.zeros(
                (d, d),
                dtype=np.uint64,
            ),

        "count_sig":
            count_sig,

        "flip_sig":
            flip_sig,

        "new_count_sig":
            np.zeros_like(
                count_sig
            ),

        "new_flip_sig":
            np.zeros_like(
                flip_sig
            ),

        "counters":
            np.zeros(
                (
                    9,
                    max_bits,
                    rep_count,
                ),
                dtype=np.uint64,
            ),

        "ages":
            np.zeros(
                n_levels,
                dtype=np.int64,
            ),

        "rng_state":
            _seed_rng(
                seed
            ),

        "alive":
            valid_mask,

        "n_lanes":
            n_lanes,
    }


# =============================================================================
# Bit unpacking
# =============================================================================

def active_lanes(
    alive,
    n_lanes,
):
    """
    Return indices of currently alive packed trajectories.
    """

    value = int(
        alive
    )

    return np.asarray(
        [
            lane
            for lane in range(
                n_lanes
            )
            if (
                value
                >> lane
            ) & 1
        ],
        dtype=np.int64,
    )


def unpack_syndrome(
    defects,
    lanes,
):
    """
    Convert packed syndrome words to shape

        (n_active, d*d)

    for PyMatching.
    """

    flat = defects.reshape(
        -1
    )

    shifts = lanes.astype(
        np.uint64
    )[:, None]

    return (
        (
            flat[None, :]
            >> shifts
        )
        & np.uint64(1)
    ).astype(
        np.uint8
    )


def actual_logicals(
    horizontal,
    vertical,
    lanes,
):
    """
    Exact logical class of each active physical error configuration.

    This uses exactly the same two cuts as the code-capacity
    _logical_mask in harrington2d_optimized.py:

        L0 = XOR_y horizontal[y,0]
        L1 = XOR_x vertical[0,x].
    """

    logical_h = np.uint64(0)
    logical_v = np.uint64(0)

    for y in range(
        horizontal.shape[0]
    ):
        logical_h ^= horizontal[
            y,
            0,
        ]

    for x in range(
        vertical.shape[1]
    ):
        logical_v ^= vertical[
            0,
            x,
        ]

    shifts = lanes.astype(
        np.uint64
    )

    out = np.empty(
        (
            len(lanes),
            2,
        ),
        dtype=np.uint8,
    )

    out[:, 0] = (
        (
            logical_h
            >> shifts
        )
        & np.uint64(1)
    ).astype(
        np.uint8
    )

    out[:, 1] = (
        (
            logical_v
            >> shifts
        )
        & np.uint64(1)
    ).astype(
        np.uint8
    )

    return out


# =============================================================================
# Advance one CA round
# =============================================================================

def advance_group(
    state,
    geometry,
    d,
    p_threshold,
    q_threshold,
):

    alive = state[
        "alive"
    ]

    if alive == 0:
        return

    # -------------------------------------------------------------------------
    # p and q noise before this CA round
    # -------------------------------------------------------------------------

    apply_noise_and_measure(
        state["horizontal"],
        state["vertical"],
        state["true_defects"],
        state["measured_defects"],
        p_threshold,
        q_threshold,
        alive,
        state["rng_state"],
        d,
    )

    # -------------------------------------------------------------------------
    # Harrington update using noisy measured syndrome
    # -------------------------------------------------------------------------

    _step(
        state["horizontal"],
        state["vertical"],
        state[
            "measured_defects"
        ],

        state["count_sig"],
        state["flip_sig"],

        state["new_count_sig"],
        state["new_flip_sig"],

        state["counters"],

        geometry[
            "site_to_rep"
        ],
        geometry[
            "roles"
        ],
        state["ages"],

        geometry[
            "U_levels"
        ],
        geometry[
            "Q_levels"
        ],
        geometry[
            "periods"
        ],
        geometry[
            "nbits_levels"
        ],

        geometry[
            "threshold_N"
        ],
        geometry[
            "threshold_C"
        ],

        alive,

        state["rng_state"],

        d,
        geometry[
            "n_levels"
        ],

        True,
    )

    # -------------------------------------------------------------------------
    # Exact syndrome AFTER the CA update for the offline checker
    # -------------------------------------------------------------------------

    _syndrome(
        state["horizontal"],
        state["vertical"],
        state["true_defects"],
        d,
    )


# =============================================================================
# Simulate one parameter point
# =============================================================================

def simulate_point(
    d,
    p,
    q,
    shots,
    rounds,
    bin_width,
    walker_chunk,
    U,
    fN,
    fC,
    seed,
):
    """
    Simulate first logical failures with PyMatching after every CA round.

    walker_chunk controls memory usage. Within one chunk, trajectories
    are stored in packed 64-lane Harrington states, while all active
    syndromes at a given time step are decoded together using
    Matching.decode_batch().
    """

    geometry = _build_geometry(
        d,
        U,
        fN,
        fC,
    )

    matching = build_matching(
        d
    )

    p_threshold = probability_threshold(
        p
    )

    q_threshold = probability_threshold(
        q
    )

    n_bins = int(
        math.ceil(
            rounds / bin_width
        )
    )

    total_at_risk = np.zeros(
        n_bins,
        dtype=np.int64,
    )

    total_failures = np.zeros(
        n_bins,
        dtype=np.int64,
    )

    total_exposure = np.zeros(
        n_bins,
        dtype=np.int64,
    )

    total_survivors_end = np.zeros(
        n_bins,
        dtype=np.int64,
    )

    completed = 0

    while completed < shots:

        chunk_shots = min(
            walker_chunk,
            shots - completed,
        )

        groups = []

        group_offset = 0

        while group_offset < chunk_shots:

            n_lanes = min(
                64,
                chunk_shots
                - group_offset,
            )

            group_id = (
                completed
                + group_offset
            ) // 64

            groups.append(
                make_group(
                    d=d,
                    n_lanes=n_lanes,
                    geometry=geometry,
                    seed=(
                        seed
                        + 1000003
                        * group_id
                    ),
                )
            )

            group_offset += (
                n_lanes
            )

        # =====================================================================
        # Time evolution of this walker chunk
        # =====================================================================

        for t in range(
            1,
            rounds + 1,
        ):

            bin_index = (
                (t - 1)
                // bin_width
            )

            # -----------------------------------------------------------------
            # Risk/exposure before the current round
            # -----------------------------------------------------------------

            alive_before = []

            total_alive_now = 0

            for state in groups:

                n_alive = int(
                    int(
                        state["alive"]
                    ).bit_count()
                )

                alive_before.append(
                    n_alive
                )

                total_alive_now += (
                    n_alive
                )

            if total_alive_now == 0:
                break

            if (
                (t - 1)
                % bin_width
                == 0
            ):
                total_at_risk[
                    bin_index
                ] += total_alive_now

            # Every trajectory alive at the beginning of this round
            # contributes one CA/QEC round of exposure.
            total_exposure[
                bin_index
            ] += total_alive_now

            # -----------------------------------------------------------------
            # Advance all packed Harrington states
            # -----------------------------------------------------------------

            syndrome_blocks = []
            logical_blocks = []
            maps = []

            for group_index, state in enumerate(
                groups
            ):

                if state[
                    "alive"
                ] == 0:
                    continue

                advance_group(
                    state=state,
                    geometry=geometry,
                    d=d,
                    p_threshold=p_threshold,
                    q_threshold=q_threshold,
                )

                lanes = active_lanes(
                    state["alive"],
                    state["n_lanes"],
                )

                if len(lanes) == 0:
                    continue

                syndrome_blocks.append(
                    unpack_syndrome(
                        state[
                            "true_defects"
                        ],
                        lanes,
                    )
                )

                logical_blocks.append(
                    actual_logicals(
                        state[
                            "horizontal"
                        ],
                        state[
                            "vertical"
                        ],
                        lanes,
                    )
                )

                maps.append(
                    (
                        group_index,
                        lanes,
                    )
                )

            if not syndrome_blocks:
                break

            syndromes = np.concatenate(
                syndrome_blocks,
                axis=0,
            )

            actual = np.concatenate(
                logical_blocks,
                axis=0,
            )

            # -----------------------------------------------------------------
            # MWPM logical prediction
            # -----------------------------------------------------------------

            predicted = matching.decode_batch(
                syndromes
            )

            predicted = np.asarray(
                predicted,
                dtype=np.uint8,
            )

            if predicted.ndim == 1:
                predicted = (
                    predicted[:, None]
                )

            if predicted.shape[1] != 2:
                raise RuntimeError(
                    "PyMatching did not return "
                    "two logical observables."
                )

            failed = np.any(
                predicted != actual,
                axis=1,
            )

            # -----------------------------------------------------------------
            # Scatter failures back to packed Harrington states
            # -----------------------------------------------------------------

            cursor = 0
            n_failed_this_round = 0

            for (
                group_index,
                lanes,
            ) in maps:

                n = len(
                    lanes
                )

                group_failed = failed[
                    cursor:
                    cursor + n
                ]

                cursor += n

                if not np.any(
                    group_failed
                ):
                    continue

                failed_word = np.uint64(
                    0
                )

                for lane in lanes[
                    group_failed
                ]:

                    failed_word |= (
                        np.uint64(1)
                        << np.uint64(
                            lane
                        )
                    )

                state = groups[
                    group_index
                ]

                state["alive"] &= (
                    ~failed_word
                )

                n_failed_this_round += int(
                    np.count_nonzero(
                        group_failed
                    )
                )

            total_failures[
                bin_index
            ] += n_failed_this_round

            # -----------------------------------------------------------------
            # Survivors at end of bin
            # -----------------------------------------------------------------

            if (
                t % bin_width == 0
                or t == rounds
            ):

                survivors = 0

                for state in groups:
                    survivors += int(
                        int(
                            state["alive"]
                        ).bit_count()
                    )

                total_survivors_end[
                    bin_index
                ] += survivors

        completed += (
            chunk_shots
        )

        print(
            f"        completed "
            f"{completed:8d}/{shots}"
        )

    # =========================================================================
    # Convert to output rows
    # =========================================================================

    rows = []

    for b in range(
        n_bins
    ):

        t0 = (
            b
            * bin_width
        )

        t1 = min(
            (b + 1)
            * bin_width,
            rounds,
        )

        if total_at_risk[b] == 0:
            break

        rows.append(
            {
                "d":
                    int(d),

                "p":
                    float(p),

                "q":
                    float(q),

                "rounds":
                    int(rounds),

                "t_start":
                    int(t0),

                "t_end":
                    int(t1),

                "shots":
                    int(shots),

                "at_risk":
                    int(
                        total_at_risk[b]
                    ),

                "failures":
                    int(
                        total_failures[b]
                    ),

                "survivors_end":
                    int(
                        total_survivors_end[b]
                    ),

                "exposure":
                    int(
                        total_exposure[b]
                    ),
            }
        )

    return rows


# =============================================================================
# Scan construction
# =============================================================================

def build_scan(
    left_distances,
    left_p,
    left_rounds,
    right_d,
    right_p_values,
    right_rounds,
):
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

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--shots",
        type=int,
        default=20_000,
    )

    parser.add_argument(
        "--walker-chunk",
        type=int,
        default=1024,
        help=(
            "Number of trajectories processed "
            "together before moving to the next chunk."
        ),
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
        default=0.002,
    )

    parser.add_argument(
        "--left-rounds",
        type=int,
        default=2500,
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
            0.0015,
            0.0020,
            0.0025,
            0.0030,
        ],
    )

    parser.add_argument(
        "--right-rounds",
        type=int,
        default=1500,
    )

    # -------------------------------------------------------------------------
    # Harrington parameters
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--U",
        type=int,
        default=16,
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
        "HARRINGTON2D PHENOMENOLOGICAL HAZARD SCAN"
    )
    print("=" * 100)

    print(
        f"shots / point      = {args.shots}"
    )
    print(
        f"walker chunk       = {args.walker_chunk}"
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
        f"points             = {len(configs)}"
    )

    print("-" * 100)

    rows = []

    for j, config in enumerate(
        configs,
        start=1,
    ):

        d = config["d"]
        p = config["p"]
        q = config["q"]
        rounds = config["rounds"]

        point_seed = (
            args.seed
            + 10007 * j
            + 101 * d
            + int(
                round(
                    1e7 * p
                )
            )
        )

        print(
            f"[{j:2d}/{len(configs):2d}] "
            f"d={d:3d}  "
            f"p=q={p:.5f}  "
            f"T={rounds:5d}  "
            f"N={args.shots:7d}"
        )

        point_rows = simulate_point(
            d=d,
            p=p,
            q=q,
            shots=args.shots,
            rounds=rounds,
            bin_width=args.bin_width,
            walker_chunk=(
                args.walker_chunk
            ),
            U=args.U,
            fN=args.fN,
            fC=args.fC,
            seed=point_seed,
        )

        rows.extend(
            point_rows
        )

    new_df = pd.DataFrame(
        rows
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
