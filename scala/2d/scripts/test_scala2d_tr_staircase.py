"""
Test whether the exact discrete SCALA failure-weight staircase w0(d)
provides a suitable predetermined reset-time schedule for SCALA2D
under phenomenological noise.

For every distance d, compute exactly

    f(w) = min_{k n = w} (k+n)

and

    w0(d) = min { w in Z_+ :
                  2 w - f(w) + 1 >= (d+1)/2 }.

The script then tests reset periods

    tR = w0(d) + delta,   delta in offsets

together with an optional long-reset control

    tR = 2 w0(d).

For each (d,p=q,tR), it estimates the stationary first-passage hazard
from two consecutive post-burn-in windows:

    h_i = F_i / E_i,

where F_i is the number of first failures and E_i is the total
surviving trajectory-time exposure in window i.

Logical failure is tested after every SCALA CA round from the exact
(noise-free) residual syndrome using an offline toric-code MWPM
checker. Measurement noise affects only the syndrome supplied to
SCALA.

Outputs
-------
1. Detailed CSV, one row per (d,p,tR)
2. Low-p log-log slope summary for every (d, staircase offset)

The goal is NOT to optimize tR separately at each p.

Instead, the diagnostic asks whether the low-p hazard scaling becomes
insensitive to tR once the exact staircase schedule is reached.
"""

from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
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
# Exact discrete staircase
# =============================================================================

def factor_sum_min(w: int) -> int:
    """
    Exact

        f(w) = min { k+n : k,n positive integers, k*n=w }.
    """

    w = int(w)

    best = None

    root = math.isqrt(w)

    for k in range(1, root + 1):

        if w % k != 0:
            continue

        n = w // k

        value = k + n

        if (
            best is None
            or value < best
        ):
            best = value

    if best is None:
        raise RuntimeError(
            f"No factorization found for w={w}"
        )

    return int(best)


def staircase_w0(d: int) -> int:
    """
    Exact discrete staircase from the manuscript:

        w0(d) = min { w in Z_+ :
                      2w - f(w) + 1 >= (d+1)/2 }.
    """

    d = int(d)

    if d <= 0 or d % 2 == 0:
        raise ValueError(
            "d must be a positive odd integer"
        )

    target = (d + 1) // 2

    w = 1

    while True:

        grown_weight = (
            2 * w
            - factor_sum_min(w)
            + 1
        )

        if grown_weight >= target:
            return w

        w += 1


def staircase_factor_pair(w: int):
    """
    Return the factor pair (k,n) attaining f(w).

    The returned pair is ordered k <= n only for reporting.
    """

    root = math.isqrt(w)

    best_pair = None
    best_sum = None

    for k in range(1, root + 1):

        if w % k != 0:
            continue

        n = w // k
        s = k + n

        if (
            best_sum is None
            or s < best_sum
        ):
            best_sum = s
            best_pair = (
                k,
                n,
            )

    return best_pair


# =============================================================================
# Probability threshold
# =============================================================================

UINT32_SCALE = 4294967296.0


def probability_threshold(p: float) -> int:

    if p <= 0.0:
        return 0

    if p >= 1.0:
        return 0x100000000

    return int(
        math.floor(
            p * UINT32_SCALE
        )
    )


# =============================================================================
# SCALA2D step with externally supplied measured syndrome
# =============================================================================

@njit
def _step_from_defect(
    h,
    v,
    N,
    W,
    E,
    S,
    defect,
    new_N,
    new_W,
    new_E,
    new_S,
    d,
):
    """
    Same SCALA2D dynamics as src.scala2d_optimized._step(), except that
    `defect` is supplied by the caller and is NOT recomputed internally.

    This permits phenomenological measurement noise:

        true syndrome -> measurement noise -> measured syndrome -> SCALA.
    """

    # -------------------------------------------------------------------------
    # Broadcast + propagate
    # -------------------------------------------------------------------------

    for y in range(d):

        yp = (
            y + 1
            if y + 1 < d
            else 0
        )

        ym = (
            y - 1
            if y > 0
            else d - 1
        )

        for x in range(d):

            xp = (
                x + 1
                if x + 1 < d
                else 0
            )

            xm = (
                x - 1
                if x > 0
                else d - 1
            )

            # North-moving signal from south.
            src_D = defect[yp, x]
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

            # South-moving signal from north.
            src_D = defect[ym, x]
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

            # West-moving signal from east.
            src_D = defect[y, xp]
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

            # East-moving signal from west.
            src_D = defect[y, xm]
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

    # -------------------------------------------------------------------------
    # Reflection / transmission
    # -------------------------------------------------------------------------

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
                ~defect[y, x]
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

    # -------------------------------------------------------------------------
    # Commit signals
    # -------------------------------------------------------------------------

    for y in range(d):

        for x in range(d):

            N[y, x] = new_N[y, x]
            W[y, x] = new_W[y, x]
            E[y, x] = new_E[y, x]
            S[y, x] = new_S[y, x]

    # -------------------------------------------------------------------------
    # Corrections
    # -------------------------------------------------------------------------

    for y in range(d):

        yp = (
            y + 1
            if y + 1 < d
            else 0
        )

        ym = (
            y - 1
            if y > 0
            else d - 1
        )

        for x in range(d):

            xp = (
                x + 1
                if x + 1 < d
                else 0
            )

            xm = (
                x - 1
                if x > 0
                else d - 1
            )

            D = defect[y, x]

            DN = defect[ym, x]
            DW = defect[y, xm]
            DE = defect[y, xp]
            DS = defect[yp, x]

            n = N[y, x]
            w = W[y, x]
            e = E[y, x]
            s = S[y, x]

            # Nearest-neighbor rule.
            move_w = D & DW
            move_n = D & ~DW & DN

            isolated = (
                D
                & ~DN
                & ~DW
                & ~DE
                & ~DS
            )

            # Exact signal-follow table.
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

            move_w |= (
                isolated
                & signal_w
            )

            move_e = (
                isolated
                & signal_e
            )

            move_n |= (
                isolated
                & signal_n
            )

            move_s = (
                isolated
                & signal_s
            )

            # Synchronous edge flips.
            h[y, x] ^= move_w
            h[y, xp] ^= move_e

            v[y, x] ^= move_n
            v[yp, x] ^= move_s


# =============================================================================
# Phenomenological round
# =============================================================================

@njit
def _advance_pheno(
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
    p_threshold,
    q_threshold,
    alive,
    rng_state,
    d,
    do_reset,
):
    """
    One complete phenomenological SCALA2D round.

    Order:

        data noise
        -> true syndrome
        -> measurement noise
        -> SCALA update
        -> optional signal reset
        -> exact post-update syndrome
    """

    # -------------------------------------------------------------------------
    # Data noise
    # -------------------------------------------------------------------------

    if p_threshold != 0:

        for y in range(d):

            for x in range(d):

                h[y, x] ^= (
                    _bernoulli_word(
                        rng_state,
                        p_threshold,
                    )
                    & alive
                )

                v[y, x] ^= (
                    _bernoulli_word(
                        rng_state,
                        p_threshold,
                    )
                    & alive
                )

    # -------------------------------------------------------------------------
    # Exact syndrome before decoder update
    # -------------------------------------------------------------------------

    _syndrome(
        h,
        v,
        true_defect,
        d,
    )

    # -------------------------------------------------------------------------
    # Measurement noise
    # -------------------------------------------------------------------------

    for y in range(d):

        for x in range(d):

            measured_defect[y, x] = (
                true_defect[y, x]
            )

            if q_threshold != 0:

                measured_defect[y, x] ^= (
                    _bernoulli_word(
                        rng_state,
                        q_threshold,
                    )
                    & alive
                )

    # -------------------------------------------------------------------------
    # SCALA update
    # -------------------------------------------------------------------------

    _step_from_defect(
        h,
        v,
        N,
        W,
        E,
        S,
        measured_defect,
        new_N,
        new_W,
        new_E,
        new_S,
        d,
    )

    # -------------------------------------------------------------------------
    # Periodic signal reset
    # -------------------------------------------------------------------------

    if do_reset:

        _clear_signals(
            N,
            W,
            E,
            S,
            d,
        )

    # -------------------------------------------------------------------------
    # Exact post-update residual syndrome
    # -------------------------------------------------------------------------

    _syndrome(
        h,
        v,
        true_defect,
        d,
    )


# =============================================================================
# Toric-code MWPM checker
# =============================================================================

def _node(
    y: int,
    x: int,
    d: int,
) -> int:

    return y * d + x


def build_matching(
    d: int,
) -> pymatching.Matching:
    """
    Matching graph for the syndrome convention used by scala2d_optimized:

        s[y,x]
          = h[y,x] ^ h[y,x+1]
            ^ v[y,x] ^ v[y+1,x].

    Observable 0:
        parity of v[0,:]

    Observable 1:
        parity of h[:,0]
    """

    matching = (
        pymatching.Matching()
    )

    # Horizontal qubits:
    #
    # h[y,x] touches syndrome vertices
    #   (y,x) and (y,x-1).
    for y in range(d):

        for x in range(d):

            xm = (
                x - 1
                if x > 0
                else d - 1
            )

            faults = (
                {1}
                if x == 0
                else set()
            )

            matching.add_edge(
                _node(
                    y,
                    x,
                    d,
                ),
                _node(
                    y,
                    xm,
                    d,
                ),
                fault_ids=faults,
                weight=1.0,
            )

    # Vertical qubits:
    #
    # v[y,x] touches syndrome vertices
    #   (y,x) and (y-1,x).
    for y in range(d):

        ym = (
            y - 1
            if y > 0
            else d - 1
        )

        for x in range(d):

            faults = (
                {0}
                if y == 0
                else set()
            )

            matching.add_edge(
                _node(
                    y,
                    x,
                    d,
                ),
                _node(
                    ym,
                    x,
                    d,
                ),
                fault_ids=faults,
                weight=1.0,
            )

    return matching


# =============================================================================
# Packed trajectory helpers
# =============================================================================

def make_group(
    d: int,
    n_lanes: int,
    seed: int,
):

    shape = (
        d,
        d,
    )

    if n_lanes == 64:

        valid_mask = MASK64

    else:

        valid_mask = np.uint64(
            (1 << n_lanes) - 1
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
            int(seed)
        ),

        "alive": valid_mask,

        "n_lanes": int(
            n_lanes
        ),
    }


def active_lanes(
    alive,
    n_lanes,
):

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
                value >> lane
            ) & 1
        ],
        dtype=np.int64,
    )


def unpack_syndrome(
    defect,
    lanes,
):

    flat = defect.reshape(
        -1
    )

    shifts = (
        lanes.astype(
            np.uint64
        )[:, None]
    )

    return (
        (
            (
                flat[None, :]
                >> shifts
            )
            & np.uint64(1)
        )
    ).astype(
        np.uint8
    )


def actual_logicals(
    h,
    v,
    lanes,
):

    logical_x = np.uint64(
        0
    )

    logical_y = np.uint64(
        0
    )

    for x in range(
        v.shape[1]
    ):

        logical_x ^= (
            v[0, x]
        )

    for y in range(
        h.shape[0]
    ):

        logical_y ^= (
            h[y, 0]
        )

    shifts = (
        lanes.astype(
            np.uint64
        )
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
            logical_x
            >> shifts
        )
        & np.uint64(1)
    ).astype(
        np.uint8
    )

    out[:, 1] = (
        (
            logical_y
            >> shifts
        )
        & np.uint64(1)
    ).astype(
        np.uint8
    )

    return out


# =============================================================================
# One Monte Carlo batch
# =============================================================================

def run_batch(
    *,
    d,
    p,
    tR,
    shots,
    burn_rounds,
    window_rounds,
    seed,
):
    """
    Run one independent batch.

    Returns:
        F1, E1, F2, E2
    """

    q = p

    p_threshold = (
        probability_threshold(
            p
        )
    )

    q_threshold = (
        probability_threshold(
            q
        )
    )

    matching = (
        build_matching(
            d
        )
    )

    groups = []

    remaining = int(
        shots
    )

    group_index = 0

    while remaining > 0:

        n_lanes = min(
            64,
            remaining,
        )

        groups.append(
            make_group(
                d=d,
                n_lanes=n_lanes,
                seed=(
                    seed
                    + 1000003
                    * group_index
                ),
            )
        )

        remaining -= n_lanes
        group_index += 1

    F1 = 0
    E1 = 0

    F2 = 0
    E2 = 0

    total_rounds = (
        burn_rounds
        + 2 * window_rounds
    )

    for t in range(
        1,
        total_rounds + 1,
    ):

        # ---------------------------------------------------------------------
        # Exposure at beginning of this first-passage step.
        # ---------------------------------------------------------------------

        alive_before = sum(
            int(
                group["alive"]
            ).bit_count()
            for group in groups
        )

        if alive_before == 0:
            break

        in_window_1 = (
            burn_rounds
            < t
            <= burn_rounds
            + window_rounds
        )

        in_window_2 = (
            burn_rounds
            + window_rounds
            < t
            <= burn_rounds
            + 2 * window_rounds
        )

        if in_window_1:
            E1 += alive_before

        elif in_window_2:
            E2 += alive_before

        # ---------------------------------------------------------------------
        # Advance all packed groups.
        # ---------------------------------------------------------------------

        for group in groups:

            if group["alive"] == 0:
                continue

            do_reset = (
                t % tR == 0
            )

            _advance_pheno(
                group["h"],
                group["v"],
                group["N"],
                group["W"],
                group["E"],
                group["S"],
                group["true_defect"],
                group["measured_defect"],
                group["new_N"],
                group["new_W"],
                group["new_E"],
                group["new_S"],
                p_threshold,
                q_threshold,
                group["alive"],
                group["rng_state"],
                d,
                do_reset,
            )

        # ---------------------------------------------------------------------
        # Gather every surviving trajectory into one MWPM batch.
        # ---------------------------------------------------------------------

        syndrome_blocks = []
        actual_blocks = []
        maps = []

        for group_index, group in enumerate(
            groups
        ):

            if group["alive"] == 0:
                continue

            lanes = active_lanes(
                group["alive"],
                group["n_lanes"],
            )

            if len(lanes) == 0:
                continue

            syndrome_blocks.append(
                unpack_syndrome(
                    group["true_defect"],
                    lanes,
                )
            )

            actual_blocks.append(
                actual_logicals(
                    group["h"],
                    group["v"],
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
            continue

        syndromes = np.concatenate(
            syndrome_blocks,
            axis=0,
        )

        actual = np.concatenate(
            actual_blocks,
            axis=0,
        )

        # ---------------------------------------------------------------------
        # Zero-syndrome fast path.
        # ---------------------------------------------------------------------

        nonzero = np.any(
            syndromes != 0,
            axis=1,
        )

        failed = np.zeros(
            len(syndromes),
            dtype=bool,
        )

        zero = ~nonzero

        if np.any(
            zero
        ):

            failed[zero] = np.any(
                actual[zero] != 0,
                axis=1,
            )

        nz = np.flatnonzero(
            nonzero
        )

        if len(nz) > 0:

            predicted = np.asarray(
                matching.decode_batch(
                    syndromes[nz]
                ),
                dtype=np.uint8,
            )

            if predicted.ndim == 1:

                predicted = (
                    predicted[:, None]
                )

            if predicted.shape[1] != 2:

                raise RuntimeError(
                    "Expected two MWPM logical observables"
                )

            failed[nz] = np.any(
                predicted
                != actual[nz],
                axis=1,
            )

        # ---------------------------------------------------------------------
        # Remove first failures.
        # ---------------------------------------------------------------------

        cursor = 0
        failures_this_round = 0

        for group_index, lanes in maps:

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

            failures_this_round += int(
                np.count_nonzero(
                    group_failed
                )
            )

            groups[
                group_index
            ]["alive"] &= (
                ~failed_word
            )

        if in_window_1:

            F1 += (
                failures_this_round
            )

        elif in_window_2:

            F2 += (
                failures_this_round
            )

    return (
        int(F1),
        int(E1),
        int(F2),
        int(E2),
    )


# =============================================================================
# Point statistics
# =============================================================================

def point_statistics(
    *,
    d,
    p,
    tR,
    min_failures,
    batch_shots,
    max_shots,
    burn_rounds,
    window_rounds,
    seed,
):
    """
    Accumulate independent batches until both windows contain the requested
    number of failures or max_shots is reached.
    """

    F1 = 0
    E1 = 0

    F2 = 0
    E2 = 0

    shots = 0
    batch_index = 0

    while (
        (
            F1 < min_failures
            or F2 < min_failures
        )
        and shots < max_shots
    ):

        n = min(
            batch_shots,
            max_shots - shots,
        )

        bF1, bE1, bF2, bE2 = (
            run_batch(
                d=d,
                p=p,
                tR=tR,
                shots=n,
                burn_rounds=(
                    burn_rounds
                ),
                window_rounds=(
                    window_rounds
                ),
                seed=(
                    seed
                    + 10000019
                    * batch_index
                ),
            )
        )

        F1 += bF1
        E1 += bE1

        F2 += bF2
        E2 += bE2

        shots += n
        batch_index += 1

    h1 = (
        F1 / E1
        if E1 > 0
        else math.nan
    )

    h2 = (
        F2 / E2
        if E2 > 0
        else math.nan
    )

    se1 = (
        math.sqrt(F1) / E1
        if (
            F1 > 0
            and E1 > 0
        )
        else math.nan
    )

    se2 = (
        math.sqrt(F2) / E2
        if (
            F2 > 0
            and E2 > 0
        )
        else math.nan
    )

    if (
        E1 + E2 > 0
    ):

        h = (
            (F1 + F2)
            / (E1 + E2)
        )

    else:

        h = math.nan

    if (
        F1 + F2 > 0
        and E1 + E2 > 0
    ):

        se = (
            math.sqrt(
                F1 + F2
            )
            / (
                E1 + E2
            )
        )

    else:

        se = math.nan

    if (
        np.isfinite(h1)
        and np.isfinite(h2)
        and np.isfinite(se1)
        and np.isfinite(se2)
        and (
            se1 * se1
            + se2 * se2
        ) > 0
    ):

        z_drift = (
            abs(
                h1 - h2
            )
            / math.sqrt(
                se1 * se1
                + se2 * se2
            )
        )

    else:

        z_drift = math.nan

    mean_h = (
        0.5
        * (
            h1 + h2
        )
        if (
            np.isfinite(h1)
            and np.isfinite(h2)
        )
        else math.nan
    )

    rel_drift = (
        abs(
            h1 - h2
        )
        / mean_h
        if (
            np.isfinite(mean_h)
            and mean_h > 0
        )
        else math.nan
    )

    enough_statistics = (
        F1 >= min_failures
        and F2 >= min_failures
    )

    stationary = (
        enough_statistics
        and np.isfinite(
            z_drift
        )
        and np.isfinite(
            rel_drift
        )
        and z_drift <= 2.0
        and rel_drift <= 0.15
    )

    return {
        "shots": int(
            shots
        ),

        "F1": int(
            F1
        ),

        "E1": int(
            E1
        ),

        "h1": float(
            h1
        ),

        "se1": float(
            se1
        ),

        "F2": int(
            F2
        ),

        "E2": int(
            E2
        ),

        "h2": float(
            h2
        ),

        "se2": float(
            se2
        ),

        "hazard": float(
            h
        ),

        "hazard_se": float(
            se
        ),

        "z_drift": float(
            z_drift
        ),

        "rel_drift": float(
            rel_drift
        ),

        "enough_statistics":
            bool(
                enough_statistics
            ),

        "stationary":
            bool(
                stationary
            ),
    }


# =============================================================================
# Worker
# =============================================================================

def run_job(job):

    d = int(
        job["d"]
    )

    p = float(
        job["p"]
    )

    tR = int(
        job["tR"]
    )

    stats = point_statistics(
        d=d,
        p=p,
        tR=tR,
        min_failures=(
            job[
                "min_failures"
            ]
        ),
        batch_shots=(
            job[
                "batch_shots"
            ]
        ),
        max_shots=(
            job[
                "max_shots"
            ]
        ),
        burn_rounds=(
            job[
                "burn_rounds"
            ]
        ),
        window_rounds=(
            job[
                "window_rounds"
            ]
        ),
        seed=(
            job[
                "seed"
            ]
        ),
    )

    return {
        **job,
        **stats,
    }


# =============================================================================
# Low-p slope summary
# =============================================================================

def write_slope_summary(
    df,
    output,
):
    """
    Fit

        log h = a + lambda log p

    separately for every distance and staircase schedule.

    This is ONLY a diagnostic for schedule stability.
    """

    rows = []

    grouped = df.groupby(
        [
            "d",
            "schedule",
            "offset",
        ],
        dropna=False,
    )

    for (
        d,
        schedule,
        offset,
    ), group in grouped:

        use = group[
            group["stationary"]
            & np.isfinite(
                group["hazard"]
            )
            & (
                group["hazard"]
                > 0
            )
            & (
                group["p"]
                > 0
            )
        ].copy()

        use = use.sort_values(
            "p"
        )

        if len(use) < 3:
            continue

        x = np.log(
            use["p"].to_numpy()
        )

        y = np.log(
            use["hazard"].to_numpy()
        )

        sigma_h = (
            use[
                "hazard_se"
            ].to_numpy()
        )

        h = (
            use[
                "hazard"
            ].to_numpy()
        )

        sigma_y = (
            sigma_h / h
        )

        valid = (
            np.isfinite(
                sigma_y
            )
            & (
                sigma_y > 0
            )
        )

        if (
            np.count_nonzero(
                valid
            )
            >= 3
        ):

            x_fit = x[
                valid
            ]

            y_fit = y[
                valid
            ]

            weights = (
                1.0
                / sigma_y[
                    valid
                ]
            )

            coeff, cov = np.polyfit(
                x_fit,
                y_fit,
                deg=1,
                w=weights,
                cov=True,
            )

            slope = float(
                coeff[0]
            )

            slope_se = float(
                math.sqrt(
                    cov[0, 0]
                )
            )

        else:

            coeff = np.polyfit(
                x,
                y,
                deg=1,
            )

            slope = float(
                coeff[0]
            )

            slope_se = math.nan

        rows.append(
            {
                "d": int(d),

                "schedule":
                    schedule,

                "offset":
                    offset,

                "n_points":
                    int(
                        len(use)
                    ),

                "p_min":
                    float(
                        use["p"].min()
                    ),

                "p_max":
                    float(
                        use["p"].max()
                    ),

                "slope_lambda":
                    slope,

                "slope_se":
                    slope_se,
            }
        )

    summary = pd.DataFrame(
        rows
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        output,
        index=False,
    )

    return summary


# =============================================================================
# CLI
# =============================================================================

def parse_args():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--d",
        nargs="+",
        type=int,
        default=[
            9,
            15,
            21,
            31,
            41,
            61,
            81,
        ],
    )

    parser.add_argument(
        "--p",
        nargs="+",
        type=float,
        default=[
            0.0015,
            0.002,
            0.003,
            0.004,
        ],
        help=(
            "Equal data and measurement rates p=q."
        ),
    )

    parser.add_argument(
        "--offsets",
        nargs="+",
        type=int,
        default=[
            -2,
            -1,
            0,
            1,
            2,
        ],
        help=(
            "Test tR = w0(d) + offset."
        ),
    )

    parser.add_argument(
        "--include-double",
        action="store_true",
        help=(
            "Also test tR = 2*w0(d)."
        ),
    )

    parser.add_argument(
        "--burn-rounds",
        type=int,
        default=200,
        help=(
            "Fixed burn-in duration. "
            "Using a common absolute burn-in avoids "
            "changing the stationarity criterion with tR."
        ),
    )

    parser.add_argument(
        "--window-rounds",
        type=int,
        default=800,
        help=(
            "Length of EACH of the two post-burn-in windows."
        ),
    )

    parser.add_argument(
        "--min-failures",
        type=int,
        default=100,
        help=(
            "Required first failures in EACH measurement window."
        ),
    )

    parser.add_argument(
        "--batch-shots",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--max-shots",
        type=int,
        default=1_000_000,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=min(
            4,
            os.cpu_count()
            or 1,
        ),
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
            "scala2d_tr_staircase_test.csv"
        ),
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():

    args = parse_args()

    jobs = []

    print()
    print("=" * 100)

    print(
        "SCALA2D EXACT-STAIRCASE RESET-TIME STABILITY TEST"
    )

    print("=" * 100)

    # -------------------------------------------------------------------------
    # Report exact staircase first.
    # -------------------------------------------------------------------------

    print()
    print(
        "Exact staircase:"
    )

    for d in args.d:

        w0 = staircase_w0(
            d
        )

        k, n = staircase_factor_pair(
            w0
        )

        print(
            f"  d={d:3d}: "
            f"w0={w0:3d}, "
            f"closest factor pair=({k},{n})"
        )

    print()

    # -------------------------------------------------------------------------
    # Jobs.
    # -------------------------------------------------------------------------

    point_index = 0

    for d in args.d:

        w0 = staircase_w0(
            d
        )

        k_star, n_star = (
            staircase_factor_pair(
                w0
            )
        )

        schedules = []

        for offset in args.offsets:

            tR = (
                w0 + offset
            )

            if tR < 1:
                continue

            schedules.append(
                (
                    f"w0{offset:+d}",
                    int(offset),
                    int(tR),
                )
            )

        if args.include_double:

            schedules.append(
                (
                    "2w0",
                    math.nan,
                    int(
                        2 * w0
                    ),
                )
            )

        # Remove accidental duplicates.
        seen = set()
        unique_schedules = []

        for item in schedules:

            if item[2] in seen:
                continue

            seen.add(
                item[2]
            )

            unique_schedules.append(
                item
            )

        for p in args.p:

            for (
                schedule,
                offset,
                tR,
            ) in unique_schedules:

                point_index += 1

                jobs.append(
                    {
                        "d": int(d),

                        "p": float(p),

                        "q": float(p),

                        "w0": int(
                            w0
                        ),

                        "k_star": int(
                            k_star
                        ),

                        "n_star": int(
                            n_star
                        ),

                        "schedule":
                            schedule,

                        "offset":
                            offset,

                        "tR": int(
                            tR
                        ),

                        "burn_rounds": int(
                            args.burn_rounds
                        ),

                        "window_rounds": int(
                            args.window_rounds
                        ),

                        "min_failures": int(
                            args.min_failures
                        ),

                        "batch_shots": int(
                            args.batch_shots
                        ),

                        "max_shots": int(
                            args.max_shots
                        ),

                        "seed": int(
                            args.seed
                            + 10000019
                            * point_index
                        ),
                    }
                )

    print(
        f"Distances       : {args.d}"
    )

    print(
        f"p=q values      : {args.p}"
    )

    print(
        f"Offsets         : {args.offsets}"
    )

    print(
        f"Include 2*w0    : {args.include_double}"
    )

    print(
        f"Burn-in         : {args.burn_rounds}"
    )

    print(
        f"Window length   : {args.window_rounds}"
    )

    print(
        f"Min failures    : {args.min_failures}"
    )

    print(
        f"Max shots/point : {args.max_shots}"
    )

    print(
        f"Workers         : {args.workers}"
    )

    print(
        f"Parameter points: {len(jobs)}"
    )

    print("=" * 100)

    rows = []

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Serial mode.
    # -------------------------------------------------------------------------

    if args.workers == 1:

        for index, job in enumerate(
            jobs,
            start=1,
        ):

            print(
                f"\n[{index}/{len(jobs)}] "
                f"d={job['d']} "
                f"p=q={job['p']:.6g} "
                f"w0={job['w0']} "
                f"tR={job['tR']} "
                f"({job['schedule']})",
                flush=True,
            )

            row = run_job(
                job
            )

            rows.append(
                row
            )

            print(
                f"    h={row['hazard']:.6e} "
                f"+- {row['hazard_se']:.2e}   "
                f"F=({row['F1']},{row['F2']})   "
                f"z={row['z_drift']:.2f}   "
                f"rel={row['rel_drift']:.3f}   "
                f"{'OK' if row['stationary'] else 'UNRESOLVED'}",
                flush=True,
            )

            pd.DataFrame(
                rows
            ).to_csv(
                args.output,
                index=False,
            )

    # -------------------------------------------------------------------------
    # Parallel parameter points.
    # -------------------------------------------------------------------------

    else:

        with ProcessPoolExecutor(
            max_workers=(
                args.workers
            )
        ) as executor:

            future_to_job = {
                executor.submit(
                    run_job,
                    job,
                ): job
                for job in jobs
            }

            done = 0

            for future in as_completed(
                future_to_job
            ):

                job = (
                    future_to_job[
                        future
                    ]
                )

                row = (
                    future.result()
                )

                rows.append(
                    row
                )

                done += 1

                print(
                    f"\n[{done}/{len(jobs)}] "
                    f"d={row['d']} "
                    f"p=q={row['p']:.6g} "
                    f"tR={row['tR']} "
                    f"({row['schedule']})",
                    flush=True,
                )

                print(
                    f"    h={row['hazard']:.6e} "
                    f"+- {row['hazard_se']:.2e}   "
                    f"F=({row['F1']},{row['F2']})   "
                    f"z={row['z_drift']:.2f}   "
                    f"rel={row['rel_drift']:.3f}   "
                    f"{'OK' if row['stationary'] else 'UNRESOLVED'}",
                    flush=True,
                )

                # -------------------------------------------------------------
                # Crash-safe-ish checkpoint.
                # Parent process is the only writer.
                # -------------------------------------------------------------

                out = (
                    pd.DataFrame(
                        rows
                    )
                    .sort_values(
                        [
                            "d",
                            "p",
                            "tR",
                        ]
                    )
                )

                tmp = (
                    args.output.with_suffix(
                        args.output.suffix
                        + ".tmp"
                    )
                )

                out.to_csv(
                    tmp,
                    index=False,
                )

                tmp.replace(
                    args.output
                )

    # =========================================================================
    # Final sort + slope summary.
    # =========================================================================

    df = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "d",
                "p",
                "tR",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    df.to_csv(
        args.output,
        index=False,
    )

    slope_output = (
        args.output.with_name(
            args.output.stem
            + "_slopes.csv"
        )
    )

    summary = (
        write_slope_summary(
            df,
            slope_output,
        )
    )

    print()
    print("=" * 100)

    print(
        f"Saved detailed data:\n"
        f"  {args.output}"
    )

    print(
        f"Saved slope summary:\n"
        f"  {slope_output}"
    )

    if len(summary):

        print()
        print(
            "Resolved low-p slopes:"
        )

        print(
            summary.to_string(
                index=False
            )
        )

    print("=" * 100)


if __name__ == "__main__":
    main()
