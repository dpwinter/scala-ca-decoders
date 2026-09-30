"""
SCALA2D phenomenological stationary-hazard scan.

This is the 2D analogue of scripts/run_pheno.py used for SCALA1D.

Protocol
--------
For every parameter point, trajectories start from the clean state and are
absorbed permanently at their FIRST logical failure.

The reset period is fixed a priori from the exact discrete SCALA staircase

    f(w) = min { k+n : k,n in Z_+, k*n=w }

    w0(d) = min { w in Z_+ :
                  2*w - f(w) + 1 >= (d+1)/2 },

and by default

    t_R(d) = w0(d).

An optional integer --reset-offset can be used for diagnostics, so that
t_R = w0(d) + reset_offset.  The production choice should use offset 0.

With B = burn_reset * t_R,

    burn-in: [0,B)
    W1:      [B,2B)
    W2:      [2B,3B)

and

    h_i = F_i / E_i,

where F_i is the number of first failures and E_i is the total surviving
trajectory-round exposure in window i.

A point is reported as resolved only if both windows have enough failures and

    |h1-h2| / sqrt(se1^2+se2^2) <= z_max

and

    |h1-h2| / ((h1+h2)/2) <= rel_max.

For accepted points,

    p_L = (F1+F2)/(E1+E2).

Noise order in every CA round
-----------------------------
1. data-qubit X noise with probability p,
2. exact syndrome,
3. independent measurement-bit noise with probability q,
4. one deterministic SCALA2D update using the measured syndrome,
5. independent noise on every stored N/W/E/S signal bit with probability p_sig,
6. scheduled signal reset if the round is a multiple of t_R,
7. exact post-update syndrome,
8. offline PyMatching logical-sector check.

Thus measurement noise affects only the syndrome supplied to SCALA.  Logical
failure is always classified from the exact residual data configuration.

Signal-noise conventions
------------------------
Default:
    p_sig = 0.

Use --signal-equals-p for the matched scan p=q=p_sig.

Alternatively, --p-sig can be supplied with one or more fixed signal-noise
probabilities.

Performance
-----------
SCALA dynamics and noise generation are bit-packed: 64 independent trajectories
are evolved in each uint64 lattice.  Logical classification is batched through
PyMatching, with a zero-syndrome fast path.  Parameter points can be run in
parallel with --workers.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import tempfile
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


BASE_SEED = 12345

DEFAULT_OUTPUT = Path(
    "data/pheno/scala2d_pheno_pL_fixed.csv"
)


# =============================================================================
# Exact discrete staircase
# =============================================================================

def factor_sum_min(w: int) -> int:
    """
    f(w) = min { k+n : k,n positive integers and k*n=w }.
    """
    w = int(w)

    if w <= 0:
        raise ValueError("w must be positive")

    best = None

    for k in range(1, math.isqrt(w) + 1):
        if w % k != 0:
            continue

        n = w // k
        value = k + n

        if best is None or value < best:
            best = value

    return int(best)


def staircase_w0(d: int) -> int:
    """
    Exact integer staircase

        w0(d) = min { w :
                      2w - f(w) + 1 >= (d+1)/2 }.
    """
    d = int(d)

    if d <= 0 or d % 2 == 0:
        raise ValueError(
            f"d must be a positive odd integer; got {d}"
        )

    target = (d + 1) // 2

    w = 1

    while True:
        if 2 * w - factor_sum_min(w) + 1 >= target:
            return w

        w += 1


# =============================================================================
# General utilities
# =============================================================================

def probability_threshold(p: float) -> int:
    p = float(p)

    if not 0.0 <= p <= 1.0:
        raise ValueError(
            f"Probability must lie in [0,1], got {p}"
        )

    if p <= 0.0:
        return 0

    if p >= 1.0:
        return 0x100000000

    return int(
        math.floor(
            p * 4294967296.0
        )
    )


def deterministic_seed(base_seed, *items):
    s = "|".join(
        [str(base_seed)]
        + [str(x) for x in items]
    )

    digest = hashlib.blake2b(
        s.encode(),
        digest_size=8,
    ).digest()

    return (
        int.from_bytes(
            digest,
            "little",
        )
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
# One SCALA2D step using an externally supplied syndrome
# =============================================================================

@njit
def scala_step_from_defect(
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
    Packed SCALA2D update equivalent to src.scala2d_optimized._step(),
    except that `defect` is supplied by the caller instead of being
    recomputed internally.  This is required for measurement noise.
    """

    # -------------------------------------------------------------------------
    # Broadcast + propagation
    # -------------------------------------------------------------------------

    for y in range(d):
        yp = y + 1 if y + 1 < d else 0
        ym = y - 1 if y > 0 else d - 1

        for x in range(d):
            xp = x + 1 if x + 1 < d else 0
            xm = x - 1 if x > 0 else d - 1

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
        yp = y + 1 if y + 1 < d else 0
        ym = y - 1 if y > 0 else d - 1

        for x in range(d):
            xp = x + 1 if x + 1 < d else 0
            xm = x - 1 if x > 0 else d - 1

            D = defect[y, x]

            DN = defect[ym, x]
            DW = defect[y, xm]
            DE = defect[y, xp]
            DS = defect[yp, x]

            n = N[y, x]
            w = W[y, x]
            e = E[y, x]
            s = S[y, x]

            # Nearest-neighbour rule: W has priority over N.
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
# One phenomenological round
# =============================================================================

@njit
def apply_round(
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
    d,
    active,
    rng_state,
    threshold_data,
    threshold_meas,
    threshold_sig,
    reset_now,
):
    """
    One complete packed phenomenological-noise CA round.
    """

    # -------------------------------------------------------------------------
    # Data noise
    # -------------------------------------------------------------------------

    if threshold_data > 0:
        for y in range(d):
            for x in range(d):
                h[y, x] ^= (
                    _bernoulli_word(
                        rng_state,
                        threshold_data,
                    )
                    & active
                )

                v[y, x] ^= (
                    _bernoulli_word(
                        rng_state,
                        threshold_data,
                    )
                    & active
                )

    # -------------------------------------------------------------------------
    # Exact syndrome + measurement noise
    # -------------------------------------------------------------------------

    _syndrome(
        h,
        v,
        true_defect,
        d,
    )

    for y in range(d):
        for x in range(d):
            measured_defect[y, x] = (
                true_defect[y, x]
            )

            if threshold_meas > 0:
                measured_defect[y, x] ^= (
                    _bernoulli_word(
                        rng_state,
                        threshold_meas,
                    )
                    & active
                )

    # -------------------------------------------------------------------------
    # Deterministic SCALA update
    # -------------------------------------------------------------------------

    scala_step_from_defect(
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
    # Signal-bit noise
    # -------------------------------------------------------------------------

    if threshold_sig > 0:
        for y in range(d):
            for x in range(d):
                N[y, x] ^= (
                    _bernoulli_word(
                        rng_state,
                        threshold_sig,
                    )
                    & active
                )

                W[y, x] ^= (
                    _bernoulli_word(
                        rng_state,
                        threshold_sig,
                    )
                    & active
                )

                E[y, x] ^= (
                    _bernoulli_word(
                        rng_state,
                        threshold_sig,
                    )
                    & active
                )

                S[y, x] ^= (
                    _bernoulli_word(
                        rng_state,
                        threshold_sig,
                    )
                    & active
                )

    # -------------------------------------------------------------------------
    # Scheduled signal reset
    # -------------------------------------------------------------------------

    if reset_now:
        _clear_signals(
            N,
            W,
            E,
            S,
            d,
        )

    # -------------------------------------------------------------------------
    # Exact post-update residual syndrome for the offline checker
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

def node(y: int, x: int, d: int) -> int:
    return y * d + x


def build_matching(d: int) -> pymatching.Matching:
    """
    Periodic matching graph for

        s[y,x] = h[y,x] ^ h[y,x+1] ^ v[y,x] ^ v[y+1,x].

    Observable 0 is parity of v[0,:].
    Observable 1 is parity of h[:,0].
    """

    matching = pymatching.Matching()

    # Horizontal qubits h[y,x] touch checks (y,x) and (y,x-1).
    for y in range(d):
        for x in range(d):
            xm = x - 1 if x > 0 else d - 1

            fault_ids = (
                {1}
                if x == 0
                else set()
            )

            matching.add_edge(
                node(y, x, d),
                node(y, xm, d),
                fault_ids=fault_ids,
                weight=1.0,
            )

    # Vertical qubits v[y,x] touch checks (y,x) and (y-1,x).
    for y in range(d):
        ym = y - 1 if y > 0 else d - 1

        for x in range(d):
            fault_ids = (
                {0}
                if y == 0
                else set()
            )

            matching.add_edge(
                node(y, x, d),
                node(ym, x, d),
                fault_ids=fault_ids,
                weight=1.0,
            )

    return matching


# =============================================================================
# Packed trajectory state
# =============================================================================

def make_group(
    d,
    n_lanes,
    seed,
):
    shape = (d, d)

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
    value = int(alive)

    return np.asarray(
        [
            lane
            for lane in range(n_lanes)
            if (value >> lane) & 1
        ],
        dtype=np.int64,
    )


def unpack_syndrome_into(
    defect,
    lanes,
    target,
):
    flat = defect.reshape(-1)

    shifts = lanes.astype(
        np.uint64
    )[:, None]

    target[:, :] = (
        (
            flat[None, :]
            >> shifts
        )
        & np.uint64(1)
    ).astype(
        np.uint8
    )


def actual_logicals_into(
    h,
    v,
    lanes,
    target,
):
    logical_x = np.uint64(0)
    logical_y = np.uint64(0)

    for x in range(v.shape[1]):
        logical_x ^= v[0, x]

    for y in range(h.shape[0]):
        logical_y ^= h[y, 0]

    shifts = lanes.astype(
        np.uint64
    )

    target[:, 0] = (
        (
            logical_x
            >> shifts
        )
        & np.uint64(1)
    ).astype(
        np.uint8
    )

    target[:, 1] = (
        (
            logical_y
            >> shifts
        )
        & np.uint64(1)
    ).astype(
        np.uint8
    )


# =============================================================================
# One independent first-passage batch
# =============================================================================

def hazard_batch(
    *,
    d,
    p,
    q,
    p_sig,
    reset_period,
    burn_reset,
    shots,
    seed,
):
    """
    Run one independent batch and return raw first-failure counts/exposures.
    """

    burn_rounds = (
        int(burn_reset)
        * int(reset_period)
    )

    total_rounds = (
        3 * burn_rounds
    )

    threshold_data = (
        probability_threshold(
            p
        )
    )

    threshold_meas = (
        probability_threshold(
            q
        )
    )

    threshold_sig = (
        probability_threshold(
            p_sig
        )
    )

    matching = build_matching(d)

    groups = []

    remaining = int(shots)
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

    # Reusable buffers.  At most `shots` trajectories can be active.
    syndrome_buffer = np.empty(
        (
            shots,
            d * d,
        ),
        dtype=np.uint8,
    )

    actual_buffer = np.empty(
        (
            shots,
            2,
        ),
        dtype=np.uint8,
    )

    map_group = np.empty(
        shots,
        dtype=np.int32,
    )

    map_lane = np.empty(
        shots,
        dtype=np.int16,
    )

    failures = np.zeros(
        2,
        dtype=np.int64,
    )

    exposure = np.zeros(
        2,
        dtype=np.int64,
    )

    for round_number in range(
        1,
        total_rounds + 1,
    ):
        if round_number <= burn_rounds:
            window = -1

        elif round_number <= 2 * burn_rounds:
            window = 0

        else:
            window = 1

        # ---------------------------------------------------------------------
        # Exposure at the start of this round
        # ---------------------------------------------------------------------

        alive_before = 0

        for group in groups:
            alive_before += int(
                group["alive"]
            ).bit_count()

        if alive_before == 0:
            break

        if window >= 0:
            exposure[window] += (
                alive_before
            )

        reset_now = (
            round_number
            % reset_period
            == 0
        )

        # ---------------------------------------------------------------------
        # Evolve all groups by one round and unpack current survivors.
        # ---------------------------------------------------------------------

        n_active = 0

        for group_index, group in enumerate(groups):
            if group["alive"] == 0:
                continue

            apply_round(
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
                d,
                group["alive"],
                group["rng_state"],
                threshold_data,
                threshold_meas,
                threshold_sig,
                reset_now,
            )

            lanes = active_lanes(
                group["alive"],
                group["n_lanes"],
            )

            n = len(lanes)

            if n == 0:
                continue

            sl = slice(
                n_active,
                n_active + n,
            )

            unpack_syndrome_into(
                group["true_defect"],
                lanes,
                syndrome_buffer[sl],
            )

            actual_logicals_into(
                group["h"],
                group["v"],
                lanes,
                actual_buffer[sl],
            )

            map_group[sl] = (
                group_index
            )

            map_lane[sl] = lanes

            n_active += n

        if n_active == 0:
            break

        syndromes = (
            syndrome_buffer[
                :n_active
            ]
        )

        actual = (
            actual_buffer[
                :n_active
            ]
        )

        # ---------------------------------------------------------------------
        # Zero-syndrome fast path.
        # ---------------------------------------------------------------------

        nonzero = np.any(
            syndromes != 0,
            axis=1,
        )

        failed = np.zeros(
            n_active,
            dtype=bool,
        )

        zero_rows = ~nonzero

        if np.any(zero_rows):
            failed[zero_rows] = np.any(
                actual[zero_rows] != 0,
                axis=1,
            )

        # ---------------------------------------------------------------------
        # MWPM only for nonzero exact residual syndromes.
        # ---------------------------------------------------------------------

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
                    "PyMatching did not return two logical observables."
                )

            failed[nz] = np.any(
                predicted
                != actual[nz],
                axis=1,
            )

        # ---------------------------------------------------------------------
        # Absorb first failures.
        # ---------------------------------------------------------------------

        failed_rows = np.flatnonzero(
            failed
        )

        failures_this_round = len(
            failed_rows
        )

        if failures_this_round > 0:
            failed_words = [
                np.uint64(0)
                for _ in groups
            ]

            for row in failed_rows:
                gi = int(
                    map_group[row]
                )

                lane = int(
                    map_lane[row]
                )

                failed_words[gi] |= (
                    np.uint64(1)
                    << np.uint64(lane)
                )

            for gi, word in enumerate(
                failed_words
            ):
                if word == 0:
                    continue

                groups[gi]["alive"] &= (
                    ~word
                )

        if window >= 0:
            failures[window] += (
                failures_this_round
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
    failures = int(failures)
    exposure = int(exposure)

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

        "stationary": bool(
            stationary
        ),

        "pL": pL,
        "pL_se": pL_se,
    }


# =============================================================================
# CSV handling
# =============================================================================

COLUMNS = [
    "d",
    "p",
    "q",
    "p_sig",

    "w0",
    "reset_offset",
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


def read_output(path):
    path = Path(path)

    if not path.exists():
        return pd.DataFrame(
            columns=COLUMNS
        )

    try:
        df = pd.read_csv(
            path
        )

    except pd.errors.EmptyDataError:
        return pd.DataFrame(
            columns=COLUMNS
        )

    for column in COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    return df


def merge_result(
    path,
    row,
):
    """
    Merge raw counts/exposures with an existing identical parameter point.
    This makes repeated invocations statistically additive.
    """

    old = read_output(
        path
    )

    key_mask = np.zeros(
        len(old),
        dtype=bool,
    )

    if len(old) > 0:
        key_mask = (
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

    if np.any(key_mask):
        idx = old.index[
            key_mask
        ][0]

        for column in [
            "shots",
            "F1",
            "E1",
            "F2",
            "E2",
        ]:
            row[column] += int(
                old.loc[
                    idx,
                    column,
                ]
            )

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
                row[
                    "_z_max"
                ]
            ),
            rel_max=(
                row[
                    "_rel_max"
                ]
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
            ~key_mask
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
    *,
    d,
    p,
    q,
    p_sig,
    reset_offset,
    burn_reset,
    shots,
    chunk_shots,
    min_window_failures,
    z_max,
    rel_max,
    seed,
):
    """
    Run one new Monte Carlo contribution for one parameter point.

    The caller writes/merges the returned row.  Keeping file I/O in the parent
    process makes --workers safe.
    """

    w0 = staircase_w0(
        d
    )

    reset_period = max(
        1,
        int(
            w0 + reset_offset
        ),
    )

    burn_rounds = (
        int(burn_reset)
        * reset_period
    )

    total_failures = np.zeros(
        2,
        dtype=np.int64,
    )

    total_exposure = np.zeros(
        2,
        dtype=np.int64,
    )

    completed = 0
    batch_index = 0

    while completed < shots:
        n = min(
            int(chunk_shots),
            int(shots - completed),
        )

        batch_seed = deterministic_seed(
            seed,
            d,
            f"{p:.12g}",
            f"{q:.12g}",
            f"{p_sig:.12g}",
            reset_period,
            burn_reset,
            batch_index,
            n,
        )

        failures, exposure = (
            hazard_batch(
                d=d,
                p=p,
                q=q,
                p_sig=p_sig,
                reset_period=(
                    reset_period
                ),
                burn_reset=(
                    burn_reset
                ),
                shots=n,
                seed=batch_seed,
            )
        )

        total_failures += (
            failures
        )

        total_exposure += (
            exposure
        )

        completed += n
        batch_index += 1

    stats = point_statistics(
        failures=(
            total_failures
        ),
        exposure=(
            total_exposure
        ),
        min_window_failures=(
            min_window_failures
        ),
        z_max=z_max,
        rel_max=rel_max,
    )

    return {
        "d": int(d),

        "p": float(p),

        "q": float(q),

        "p_sig": float(
            p_sig
        ),

        "w0": int(
            w0
        ),

        "reset_offset": int(
            reset_offset
        ),

        "reset_period": int(
            reset_period
        ),

        "burn_reset": int(
            burn_reset
        ),

        "burn_rounds": int(
            burn_rounds
        ),

        "shots": int(
            shots
        ),

        "F1": int(
            stats["F1"]
        ),

        "E1": int(
            stats["E1"]
        ),

        "h1": float(
            stats["h1"]
        ),

        "se1": float(
            stats["se1"]
        ),

        "F2": int(
            stats["F2"]
        ),

        "E2": int(
            stats["E2"]
        ),

        "h2": float(
            stats["h2"]
        ),

        "se2": float(
            stats["se2"]
        ),

        "z_drift": float(
            stats["z"]
        ),

        "rel_drift": float(
            stats["rel"]
        ),

        "stationary": bool(
            stats["stationary"]
        ),

        "pL": float(
            stats["pL"]
        ),

        "pL_se": float(
            stats["pL_se"]
        ),

        "_min_window_failures": int(
            min_window_failures
        ),

        "_z_max": float(
            z_max
        ),

        "_rel_max": float(
            rel_max
        ),
    }


def run_point_worker(job):
    return run_point(
        **job
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
        help=(
            "Measurement-noise rates. "
            "Default: q=p point by point."
        ),
    )

    parser.add_argument(
        "--p-sig",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Fixed signal-noise rates to combine with every (p,q). "
            "Default: p_sig=0."
        ),
    )

    parser.add_argument(
        "--signal-equals-p",
        action="store_true",
        help=(
            "Use p_sig=p point by point. "
            "Cannot be combined with --p-sig."
        ),
    )

    parser.add_argument(
        "--reset-offset",
        type=int,
        default=0,
        help=(
            "Use t_R=w0(d)+offset. "
            "Production staircase schedule is offset 0."
        ),
    )

    parser.add_argument(
        "--burn-reset",
        type=int,
        default=10,
        help=(
            "Burn-in AND each measurement-window length "
            "in reset periods."
        ),
    )

    parser.add_argument(
        "--shots",
        type=int,
        default=100000,
        help=(
            "New trajectories generated per parameter point "
            "during this invocation."
        ),
    )

    parser.add_argument(
        "--chunk-shots",
        type=int,
        default=4096,
        help=(
            "Trajectories held in memory at once per parameter point."
        ),
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
        "--workers",
        type=int,
        default=1,
        help=(
            "Independent parameter points simulated concurrently."
        ),
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


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    if (
        args.signal_equals_p
        and args.p_sig is not None
    ):
        raise ValueError(
            "--signal-equals-p and --p-sig are mutually exclusive."
        )

    if args.workers < 1:
        raise ValueError(
            "--workers must be >= 1"
        )

    if args.chunk_shots < 1:
        raise ValueError(
            "--chunk-shots must be positive"
        )

    for d in args.d:
        if d <= 0 or d % 2 == 0:
            raise ValueError(
                f"d must be positive odd; got {d}"
            )

    for value in args.p:
        probability_threshold(
            value
        )

    if args.q is not None:
        for value in args.q:
            probability_threshold(
                value
            )

    if args.p_sig is not None:
        for value in args.p_sig:
            probability_threshold(
                value
            )

    print()
    print("=" * 100)

    print(
        "SCALA2D PHENOMENOLOGICAL FIXED-WINDOW "
        "FIRST-PASSAGE HAZARD"
    )

    print("=" * 100)

    print(
        "reset schedule       = "
        "exact discrete staircase "
        "t_R=w0(d)"
        + (
            f"{args.reset_offset:+d}"
            if args.reset_offset != 0
            else ""
        )
    )

    print(
        f"burn/window length   = "
        f"{args.burn_reset} t_R"
    )

    print(
        f"new shots / point    = "
        f"{args.shots}"
    )

    print(
        f"chunk shots          = "
        f"{args.chunk_shots}"
    )

    print(
        f"workers              = "
        f"{args.workers}"
    )

    print(
        f"output               = "
        f"{args.output}"
    )

    jobs = []

    point_index = 0

    for d in args.d:
        w0 = staircase_w0(
            d
        )

        reset_period = max(
            1,
            w0 + args.reset_offset,
        )

        print(
            f"  d={d:3d}: "
            f"w0={w0:3d}, "
            f"t_R={reset_period:3d}"
        )

        for p in args.p:
            if args.q is None:
                q_values = [
                    float(p)
                ]

            else:
                q_values = [
                    float(q)
                    for q in args.q
                ]

            if args.signal_equals_p:
                p_sig_values = [
                    float(p)
                ]

            elif args.p_sig is None:
                p_sig_values = [
                    0.0
                ]

            else:
                p_sig_values = [
                    float(value)
                    for value in args.p_sig
                ]

            for q in q_values:
                for p_sig in p_sig_values:
                    point_index += 1

                    jobs.append(
                        {
                            "d": int(d),
                            "p": float(p),
                            "q": float(q),
                            "p_sig": float(
                                p_sig
                            ),
                            "reset_offset": int(
                                args.reset_offset
                            ),
                            "burn_reset": int(
                                args.burn_reset
                            ),
                            "shots": int(
                                args.shots
                            ),
                            "chunk_shots": int(
                                args.chunk_shots
                            ),
                            "min_window_failures": int(
                                args.min_window_failures
                            ),
                            "z_max": float(
                                args.z_max
                            ),
                            "rel_max": float(
                                args.rel_max
                            ),
                            "seed": deterministic_seed(
                                args.seed,
                                "point",
                                point_index,
                                d,
                                f"{p:.12g}",
                                f"{q:.12g}",
                                f"{p_sig:.12g}",
                                reset_period,
                            ),
                        }
                    )

    print(
        f"parameter points     = "
        f"{len(jobs)}"
    )

    print("=" * 100)

    rows = []

    # -------------------------------------------------------------------------
    # Serial mode
    # -------------------------------------------------------------------------

    if args.workers == 1:
        for index, job in enumerate(
            jobs,
            start=1,
        ):
            w0 = staircase_w0(
                job["d"]
            )

            tR = max(
                1,
                w0 + job["reset_offset"],
            )

            B = (
                job["burn_reset"]
                * tR
            )

            print()
            print(
                f"[{index}/{len(jobs)}] "
                f"d={job['d']:3d}  "
                f"p={job['p']:.6f}  "
                f"q={job['q']:.6f}  "
                f"p_sig={job['p_sig']:.6f}  "
                f"w0={w0:3d}  "
                f"tR={tR:3d}  "
                f"B={B:5d}  "
                f"T={3*B:5d}"
            )

            row = run_point_worker(
                job
            )

            rows.append(
                row
            )

            print(
                f"    W1: "
                f"F={row['F1']:7d}  "
                f"E={row['E1']:12d}  "
                f"h={row['h1']:.6e}"
            )

            print(
                f"    W2: "
                f"F={row['F2']:7d}  "
                f"E={row['E2']:12d}  "
                f"h={row['h2']:.6e}"
            )

            print(
                f"    drift: "
                f"z={row['z_drift']:.2f}  "
                f"rel={row['rel_drift']:.3f}  "
                f"{'OK' if row['stationary'] else 'UNRESOLVED'}"
            )

            if row["stationary"]:
                print(
                    f"    pL={row['pL']:.6e} "
                    f"+- {row['pL_se']:.2e}"
                )

            merge_result(
                args.output,
                row,
            )

    # -------------------------------------------------------------------------
    # Parallel mode
    # -------------------------------------------------------------------------

    else:
        with ProcessPoolExecutor(
            max_workers=(
                args.workers
            )
        ) as executor:
            future_to_job = {
                executor.submit(
                    run_point_worker,
                    job,
                ): job
                for job in jobs
            }

            completed = 0

            for future in as_completed(
                future_to_job
            ):
                job = future_to_job[
                    future
                ]

                row = future.result()

                completed += 1
                rows.append(
                    row
                )

                print()
                print(
                    f"[{completed}/{len(jobs)}] "
                    f"d={row['d']:3d}  "
                    f"p={row['p']:.6f}  "
                    f"q={row['q']:.6f}  "
                    f"p_sig={row['p_sig']:.6f}  "
                    f"tR={row['reset_period']:3d}"
                )

                print(
                    f"    W1: "
                    f"F={row['F1']:7d}  "
                    f"E={row['E1']:12d}  "
                    f"h={row['h1']:.6e}"
                )

                print(
                    f"    W2: "
                    f"F={row['F2']:7d}  "
                    f"E={row['E2']:12d}  "
                    f"h={row['h2']:.6e}"
                )

                print(
                    f"    drift: "
                    f"z={row['z_drift']:.2f}  "
                    f"rel={row['rel_drift']:.3f}  "
                    f"{'OK' if row['stationary'] else 'UNRESOLVED'}"
                )

                if row["stationary"]:
                    print(
                        f"    pL={row['pL']:.6e} "
                        f"+- {row['pL_se']:.2e}"
                    )

                # Only the parent writes the CSV.
                merge_result(
                    args.output,
                    row,
                )

    print()
    print("=" * 100)

    print(
        f"Saved: {args.output}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
