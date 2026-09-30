from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pymatching
from numba import njit

from src.scala2d_optimized import (
    MASK64,
    _bernoulli_word,
    _clear_signals,
    _seed_rng,
    _syndrome,
)


# =============================================================================
# Probability helper
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
# SCALA2D phenomenological step
# =============================================================================

@njit
def step_pheno(
    h,
    v,
    N,
    W,
    E,
    S,
    true_defect,
    measured_defect,
    new_N,
    new_W,
    new_E,
    new_S,
    q_threshold,
    alive,
    rng_state,
    d,
):
    # Exact syndrome.
    _syndrome(
        h,
        v,
        true_defect,
        d,
    )

    # Measurement noise.
    for y in range(d):
        for x in range(d):
            measured_defect[y, x] = true_defect[y, x]

            if q_threshold != 0:
                measured_defect[y, x] ^= (
                    _bernoulli_word(
                        rng_state,
                        q_threshold,
                    )
                    & alive
                )

    # Broadcast / propagation.
    for y in range(d):
        yp = (y + 1) % d
        ym = (y - 1) % d

        for x in range(d):
            xp = (x + 1) % d
            xm = (x - 1) % d

            src_D = measured_defect[yp, x]
            src_N = N[yp, x]
            src_S = S[yp, x]

            new_N[y, x] = (
                src_N
                | (
                    src_D
                    & ~src_N
                    & ~src_S
                )
            )

            src_D = measured_defect[ym, x]
            src_N = N[ym, x]
            src_S = S[ym, x]

            new_S[y, x] = (
                src_S
                | (
                    src_D
                    & ~src_N
                    & ~src_S
                )
            )

            src_D = measured_defect[y, xp]
            src_W = W[y, xp]
            src_E = E[y, xp]

            new_W[y, x] = (
                src_W
                | (
                    src_D
                    & ~src_W
                    & ~src_E
                )
            )

            src_D = measured_defect[y, xm]
            src_W = W[y, xm]
            src_E = E[y, xm]

            new_E[y, x] = (
                src_E
                | (
                    src_D
                    & ~src_W
                    & ~src_E
                )
            )

    # Reflection / transmission.
    for y in range(d):
        for x in range(d):
            n = new_N[y, x]
            w = new_W[y, x]
            e = new_E[y, x]
            s = new_S[y, x]

            at_least_two = (
                (n & w)
                | (n & e)
                | (n & s)
                | (w & e)
                | (w & s)
                | (e & s)
            )

            reflect = (
                ~measured_defect[y, x]
                & at_least_two
            )

            keep = ~reflect

            new_N[y, x] = (
                (keep & n)
                | (reflect & s)
            )

            new_W[y, x] = (
                (keep & w)
                | (reflect & e)
            )

            new_E[y, x] = (
                (keep & e)
                | (reflect & w)
            )

            new_S[y, x] = (
                (keep & s)
                | (reflect & n)
            )

    # Commit signals.
    for y in range(d):
        for x in range(d):
            N[y, x] = new_N[y, x]
            W[y, x] = new_W[y, x]
            E[y, x] = new_E[y, x]
            S[y, x] = new_S[y, x]

    # Corrections.
    for y in range(d):
        yp = (y + 1) % d
        ym = (y - 1) % d

        for x in range(d):
            xp = (x + 1) % d
            xm = (x - 1) % d

            D = measured_defect[y, x]

            DN = measured_defect[ym, x]
            DW = measured_defect[y, xm]
            DE = measured_defect[y, xp]
            DS = measured_defect[yp, x]

            n = N[y, x]
            w = W[y, x]
            e = E[y, x]
            s = S[y, x]

            move_w = D & DW
            move_n = D & ~DW & DN

            isolated = (
                D
                & ~DN
                & ~DW
                & ~DE
                & ~DS
            )

            signal_w = (
                (e & ~n & ~w & ~s)
                | (n & e & ~w & ~s)
                | (n & e & s & ~w)
            )

            signal_e = (
                (w & ~n & ~e & ~s)
                | (n & w & s & ~e)
            )

            signal_n = (
                (s & ~n & ~w & ~e)
                | (w & e & s & ~n)
            )

            signal_s = (
                (n & ~w & ~e & ~s)
                | (n & w & ~e & ~s)
                | (n & w & e & ~s)
            )

            move_w |= isolated & signal_w
            move_e = isolated & signal_e
            move_n |= isolated & signal_n
            move_s = isolated & signal_s

            h[y, x] ^= move_w
            h[y, xp] ^= move_e

            v[y, x] ^= move_n
            v[yp, x] ^= move_s


# =============================================================================
# Data noise
# =============================================================================

@njit
def apply_data_noise(
    h,
    v,
    threshold,
    alive,
    rng_state,
    d,
):
    if threshold == 0:
        return

    for y in range(d):
        for x in range(d):
            h[y, x] ^= (
                _bernoulli_word(
                    rng_state,
                    threshold,
                )
                & alive
            )

            v[y, x] ^= (
                _bernoulli_word(
                    rng_state,
                    threshold,
                )
                & alive
            )


# =============================================================================
# MWPM
# =============================================================================

def node(y, x, d):
    return y * d + x


def build_matching(d):
    matching = pymatching.Matching()

    for y in range(d):
        for x in range(d):
            xm = (x - 1) % d

            matching.add_edge(
                node(y, x, d),
                node(y, xm, d),
                fault_ids=(
                    {1}
                    if x == 0
                    else set()
                ),
                weight=1.0,
            )

    for y in range(d):
        ym = (y - 1) % d

        for x in range(d):
            matching.add_edge(
                node(y, x, d),
                node(ym, x, d),
                fault_ids=(
                    {0}
                    if y == 0
                    else set()
                ),
                weight=1.0,
            )

    return matching


def active_lanes(
    alive,
    n_lanes,
):
    value = int(alive)

    return np.asarray(
        [
            lane
            for lane in range(n_lanes)
            if (value >> lane) & 1
        ],
        dtype=np.int64,
    )


def unpack_syndrome(
    defect,
    lanes,
):
    flat = defect.reshape(-1)

    shifts = lanes.astype(
        np.uint64
    )[:, None]

    return (
        (
            flat[None, :]
            >> shifts
        )
        & np.uint64(1)
    ).astype(np.uint8)


def actual_logicals(
    h,
    v,
    lanes,
):
    logical_x = np.uint64(0)
    logical_y = np.uint64(0)

    d = h.shape[0]

    for x in range(d):
        logical_x ^= v[0, x]

    for y in range(d):
        logical_y ^= h[y, 0]

    shifts = lanes.astype(
        np.uint64
    )

    out = np.empty(
        (len(lanes), 2),
        dtype=np.uint8,
    )

    out[:, 0] = (
        (
            logical_x
            >> shifts
        )
        & np.uint64(1)
    ).astype(np.uint8)

    out[:, 1] = (
        (
            logical_y
            >> shifts
        )
        & np.uint64(1)
    ).astype(np.uint8)

    return out


# =============================================================================
# State
# =============================================================================

def make_group(
    d,
    n_lanes,
    seed,
):
    shape = (d, d)

    valid_mask = (
        MASK64
        if n_lanes == 64
        else np.uint64(
            (1 << n_lanes) - 1
        )
    )

    return {
        "h": np.zeros(
            shape,
            dtype=np.uint64,
        ),
        "v": np.zeros(
            shape,
            dtype=np.uint64,
        ),

        "N": np.zeros(
            shape,
            dtype=np.uint64,
        ),
        "W": np.zeros(
            shape,
            dtype=np.uint64,
        ),
        "E": np.zeros(
            shape,
            dtype=np.uint64,
        ),
        "S": np.zeros(
            shape,
            dtype=np.uint64,
        ),

        "true_defect": np.zeros(
            shape,
            dtype=np.uint64,
        ),
        "measured_defect": np.zeros(
            shape,
            dtype=np.uint64,
        ),

        "new_N": np.zeros(
            shape,
            dtype=np.uint64,
        ),
        "new_W": np.zeros(
            shape,
            dtype=np.uint64,
        ),
        "new_E": np.zeros(
            shape,
            dtype=np.uint64,
        ),
        "new_S": np.zeros(
            shape,
            dtype=np.uint64,
        ),

        "rng_state": _seed_rng(
            seed
        ),
        "alive": valid_mask,
        "n_lanes": n_lanes,
        "reset_clock": 0,
    }


# =============================================================================
# One point
# =============================================================================

def simulate_point(
    d,
    p,
    q,
    reset_factor,
    shots,
    rounds,
    block_width,
    seed,
):
    matching = build_matching(d)

    tR = max(
        1,
        int(
            round(
                reset_factor * d
            )
        ),
    )

    p_threshold = probability_threshold(
        p
    )

    q_threshold = probability_threshold(
        q
    )

    n_blocks = int(
        math.ceil(
            rounds / block_width
        )
    )

    failures = np.zeros(
        n_blocks,
        dtype=np.int64,
    )

    survivors_end = np.zeros(
        n_blocks,
        dtype=np.int64,
    )

    sum_failure_t = np.zeros(
        n_blocks,
        dtype=np.float64,
    )

    sum_failure_t2 = np.zeros(
        n_blocks,
        dtype=np.float64,
    )

    groups = []

    offset = 0

    while offset < shots:
        n_lanes = min(
            64,
            shots - offset,
        )

        groups.append(
            make_group(
                d=d,
                n_lanes=n_lanes,
                seed=(
                    seed
                    + 1000003
                    * len(groups)
                ),
            )
        )

        offset += n_lanes

    for t in range(
        1,
        rounds + 1,
    ):
        syndrome_blocks = []
        logical_blocks = []
        maps = []

        any_alive = False

        for gi, state in enumerate(groups):

            if state["alive"] == 0:
                continue

            any_alive = True

            apply_data_noise(
                state["h"],
                state["v"],
                p_threshold,
                state["alive"],
                state["rng_state"],
                d,
            )

            step_pheno(
                state["h"],
                state["v"],
                state["N"],
                state["W"],
                state["E"],
                state["S"],
                state["true_defect"],
                state["measured_defect"],
                state["new_N"],
                state["new_W"],
                state["new_E"],
                state["new_S"],
                q_threshold,
                state["alive"],
                state["rng_state"],
                d,
            )

            state["reset_clock"] += 1

            if state["reset_clock"] >= tR:
                _clear_signals(
                    state["N"],
                    state["W"],
                    state["E"],
                    state["S"],
                    d,
                )

                state["reset_clock"] = 0

            _syndrome(
                state["h"],
                state["v"],
                state["true_defect"],
                d,
            )

            lanes = active_lanes(
                state["alive"],
                state["n_lanes"],
            )

            if len(lanes) == 0:
                continue

            syndrome_blocks.append(
                unpack_syndrome(
                    state["true_defect"],
                    lanes,
                )
            )

            logical_blocks.append(
                actual_logicals(
                    state["h"],
                    state["v"],
                    lanes,
                )
            )

            maps.append(
                (
                    gi,
                    lanes,
                )
            )

        if not any_alive:
            break

        syndromes = np.concatenate(
            syndrome_blocks,
            axis=0,
        )

        actual = np.concatenate(
            logical_blocks,
            axis=0,
        )

        predicted = np.asarray(
            matching.decode_batch(
                syndromes
            ),
            dtype=np.uint8,
        )

        if predicted.ndim == 1:
            predicted = (
                predicted[:, None]
            )

        failed = np.any(
            predicted != actual,
            axis=1,
        )

        cursor = 0

        block = (
            (t - 1)
            // block_width
        )

        for gi, lanes in maps:
            n = len(lanes)

            local_failed = failed[
                cursor:
                cursor + n
            ]

            cursor += n

            if not np.any(
                local_failed
            ):
                continue

            failed_word = np.uint64(0)

            failed_lanes = lanes[
                local_failed
            ]

            for lane in failed_lanes:
                failed_word |= (
                    np.uint64(1)
                    << np.uint64(lane)
                )

            groups[gi]["alive"] &= (
                ~failed_word
            )

            nf = len(
                failed_lanes
            )

            failures[block] += nf

            sum_failure_t[block] += (
                nf * t
            )

            sum_failure_t2[block] += (
                nf * t * t
            )

        if (
            t % block_width == 0
            or t == rounds
        ):
            survivors = 0

            for state in groups:
                survivors += int(
                    int(
                        state["alive"]
                    ).bit_count()
                )

            survivors_end[
                block
            ] = survivors

    rows = []

    for b in range(
        n_blocks
    ):
        t0 = (
            b * block_width
        )

        t1 = min(
            (b + 1)
            * block_width,
            rounds,
        )

        rows.append(
            {
                "d": int(d),
                "p": float(p),
                "p_meas": float(q),
                "reset_factor":
                    float(
                        reset_factor
                    ),
                "reset_period":
                    int(tR),
                "rounds":
                    int(rounds),
                "block":
                    int(b),
                "t_start":
                    int(t0),
                "t_end":
                    int(t1),
                "shots":
                    int(shots),
                "failures":
                    int(
                        failures[b]
                    ),
                "survivors_end":
                    int(
                        survivors_end[b]
                    ),
                "sum_failure_t":
                    float(
                        sum_failure_t[b]
                    ),
                "sum_failure_t2":
                    float(
                        sum_failure_t2[b]
                    ),
            }
        )

    return rows


# =============================================================================
# CLI
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--d",
        nargs="+",
        type=int,
        default=[
            9,
            21,
            41,
            61,
        ],
    )

    parser.add_argument(
        "--p",
        type=float,
        default=0.008,
    )

    parser.add_argument(
        "--reset-factor",
        nargs="+",
        type=float,
        default=[
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
        ],
    )

    parser.add_argument(
        "--shots",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--rounds-multiplier",
        type=int,
        default=120,
        help=(
            "T_max = multiplier * d"
        ),
    )

    parser.add_argument(
        "--block-width",
        type=int,
        default=50,
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
            "pheno_tR_lifetime_Tmax120d.csv"
        ),
    )

    args = parser.parse_args()

    rows = []

    points = (
        len(args.d)
        * len(
            args.reset_factor
        )
    )

    k = 0

    for d in args.d:

        rounds = (
            args.rounds_multiplier
            * d
        )

        for alpha in args.reset_factor:

            k += 1

            seed = (
                args.seed
                + 10007 * k
                + 101 * d
                + int(
                    1e5 * alpha
                )
            )

            print(
                f"[{k:2d}/{points:2d}] "
                f"d={d:3d} "
                f"p=q={args.p:.5f} "
                f"alpha={alpha:.2f} "
                f"T={rounds:6d} "
                f"N={args.shots:7d}"
            )

            point_rows = (
                simulate_point(
                    d=d,
                    p=args.p,
                    q=args.p,
                    reset_factor=alpha,
                    shots=args.shots,
                    rounds=rounds,
                    block_width=(
                        args.block_width
                    ),
                    seed=seed,
                )
            )

            rows.extend(
                point_rows
            )

    out = pd.DataFrame(
        rows
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        args.output,
        index=False,
    )

    print()
    print(
        f"Saved: {args.output}"
    )


if __name__ == "__main__":
    main()
