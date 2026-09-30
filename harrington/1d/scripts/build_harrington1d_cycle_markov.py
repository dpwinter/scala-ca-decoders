from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from numba import njit

from src.harrington1d_optimized import (
    MASK64,
    _bernoulli_word,
    _build_geometry,
    _seed_rng,
    _step,
)


D = 9
FAIL_WEIGHT = D // 2


# =============================================================================
# State space
# =============================================================================

def transient_states():
    """
    All physical error configurations which have not yet failed.

    For d=9:
        failure <=> Hamming weight > 4.
    """

    states = [
        x
        for x in range(1 << D)
        if x.bit_count() <= FAIL_WEIGHT
    ]

    index = {
        state: i
        for i, state in enumerate(states)
    }

    return states, index


# =============================================================================
# Probability threshold
# =============================================================================

def threshold(p: float) -> int:

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
# Packed-state helpers
# =============================================================================

@njit(inline="always")
def majority_failure_mask(
    qubits,
    valid,
):
    """
    Return lanes whose instantaneous physical error weight exceeds 4.
    """

    failed = np.uint64(0)

    for lane in range(64):

        bit = (
            np.uint64(1)
            << np.uint64(lane)
        )

        if (valid & bit) == 0:
            continue

        weight = 0

        for i in range(D):

            if qubits[i] & bit:
                weight += 1

        if weight > FAIL_WEIGHT:
            failed |= bit

    return failed


@njit(inline="always")
def configuration_for_lane(
    qubits,
    lane,
):
    """
    Convert one packed trajectory back to a 9-bit integer.
    """

    bit = (
        np.uint64(1)
        << np.uint64(lane)
    )

    state = 0

    for i in range(D):

        if qubits[i] & bit:
            state |= 1 << i

    return state


@njit
def initialise_qubits(
    qubits,
    physical_state,
    valid,
):
    """
    Put every packed lane into the same physical error configuration.
    """

    for i in range(D):

        if (
            physical_state
            >> i
        ) & 1:

            qubits[i] = valid

        else:

            qubits[i] = np.uint64(0)


# =============================================================================
# One exact hierarchy cycle
# =============================================================================

@njit
def simulate_cycle_batch(
    physical_state,
    walkers,
    p_threshold,
    q_threshold,
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
    """
    Simulate one complete level-1 Harrington cycle for <=64 walkers.

    Start immediately after a level-1 release:
        * data error configuration = physical_state
        * all signal fields = zero
        * all counters = zero
        * age = period - 1

    Evolve exactly period steps.

    Failure is checked after EVERY CA step.

    Returns:
        final_state[lane] = 0..511 for survivor
                          = -1 for logical failure
    """

    if walkers == 64:

        valid = MASK64

    else:

        valid = (
            (
                np.uint64(1)
                << np.uint64(walkers)
            )
            - np.uint64(1)
        )

    qubits = np.zeros(
        D,
        dtype=np.uint64,
    )

    initialise_qubits(
        qubits,
        physical_state,
        valid,
    )

    defects = np.zeros(
        D,
        dtype=np.uint64,
    )

    count_L = np.zeros(
        (n_levels, D),
        dtype=np.uint64,
    )

    count_R = np.zeros_like(
        count_L
    )

    flip_L = np.zeros_like(
        count_L
    )

    flip_R = np.zeros_like(
        count_L
    )

    new_count_L = np.zeros_like(
        count_L
    )

    new_count_R = np.zeros_like(
        count_L
    )

    new_flip_L = np.zeros_like(
        count_L
    )

    new_flip_R = np.zeros_like(
        count_L
    )

    counters = np.zeros(
        (
            3,
            max_bits,
            rep_count,
        ),
        dtype=np.uint64,
    )

    ages = np.empty(
        n_levels,
        dtype=np.int64,
    )

    # ---------------------------------------------------------
    # Exact post-release phase.
    #
    # _step() increments age before doing anything else.
    # Starting at period-1 therefore makes the first new step age=0.
    # ---------------------------------------------------------

    for level in range(n_levels):

        ages[level] = (
            periods[level]
            - 1
        )

    alive = valid

    tau = periods[0]

    for _ in range(tau):

        if alive == 0:
            break

        # =====================================================
        # 1. Data noise
        # =====================================================

        if p_threshold != 0:

            for i in range(D):

                qubits[i] ^= (
                    _bernoulli_word(
                        rng_state,
                        p_threshold,
                    )
                    & alive
                )

        # =====================================================
        # 2. True syndrome + measurement noise
        # =====================================================

        for i in range(D):

            right = (
                i + 1
                if i + 1 < D
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

        # =====================================================
        # 3. Harrington CA update
        # =====================================================

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

            D,
            n_levels,

            True,           # syndrome already contains q noise
        )

        # =====================================================
        # 4. Logical failure after THIS CA step
        # =====================================================

        failed = (
            majority_failure_mask(
                qubits,
                alive,
            )
            & alive
        )

        alive &= ~failed

    # =========================================================
    # Read out survivors
    # =========================================================

    result = np.full(
        walkers,
        -1,
        dtype=np.int64,
    )

    for lane in range(walkers):

        bit = (
            np.uint64(1)
            << np.uint64(lane)
        )

        if alive & bit:

            result[lane] = (
                configuration_for_lane(
                    qubits,
                    lane,
                )
            )

    return result


# =============================================================================
# Estimate one row of K_tau
# =============================================================================

def estimate_row(
    physical_state,
    state_index,
    shots,
    p,
    q,
    seed,
    geometry,
):
    counts = np.zeros(
        len(state_index),
        dtype=np.int64,
    )

    failures = 0

    p_threshold = threshold(p)
    q_threshold = threshold(q)

    done = 0
    batch_index = 0

    while done < shots:

        walkers = min(
            64,
            shots - done,
        )

        rng_state = _seed_rng(
            seed
            + 1000003
            * batch_index
        )

        result = simulate_cycle_batch(
            physical_state,
            walkers,
            p_threshold,
            q_threshold,
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

        for value in result:

            if value < 0:

                failures += 1

            else:

                # Every surviving configuration must remain transient.
                j = state_index.get(
                    int(value)
                )

                if j is None:

                    raise RuntimeError(
                        "Surviving cycle ended in a majority-failed state."
                    )

                counts[j] += 1

        done += walkers
        batch_index += 1

    return counts, failures


# =============================================================================
# Full cycle-boundary transition matrix
# =============================================================================

def build_matrix(
    p,
    q,
    shots_per_state,
    seed,
):

    states, state_index = (
        transient_states()
    )

    n = len(states)

    assert n == 256

    geometry = _build_geometry(
        D,
        10,     # U
        0.4,    # fN
        0.9,    # fC
    )

    if geometry["n_levels"] != 1:

        raise RuntimeError(
            "This script is specifically for d=9."
        )

    tau = int(
        geometry["periods"][0]
    )

    print(
        f"d={D}, transient states={n}, tau={tau}"
    )

    K = np.zeros(
        (n, n),
        dtype=np.float64,
    )

    p_absorb = np.zeros(
        n,
        dtype=np.float64,
    )

    for i, state in enumerate(
        states
    ):

        counts, failures = (
            estimate_row(
                physical_state=state,
                state_index=state_index,
                shots=shots_per_state,
                p=p,
                q=q,
                seed=(
                    seed
                    + 10000019 * i
                ),
                geometry=geometry,
            )
        )

        K[i, :] = (
            counts
            / shots_per_state
        )

        p_absorb[i] = (
            failures
            / shots_per_state
        )

        if (
            i % 16 == 0
            or i == n - 1
        ):

            print(
                f"    row {i + 1:3d}/{n}: "
                f"state={state:03x} "
                f"weight={state.bit_count()} "
                f"Pabs={p_absorb[i]:.4e}"
            )

    return (
        states,
        K,
        p_absorb,
        tau,
    )


# =============================================================================
# Spectral analysis
# =============================================================================

def analyse_matrix(
    K,
    tau,
):

    eigvals = np.linalg.eigvals(
        K
    )

    rho = float(
        np.max(
            np.abs(
                eigvals
            )
        )
    )

    h_cycle = (
        1.0 - rho
    )

    # Equivalent exponential/geometric loss per elementary CA step.
    h_step = (
        1.0
        - rho**(
            1.0 / tau
        )
    )

    mean_cycle_lifetime = None

    try:

        e0 = np.zeros(
            K.shape[0],
            dtype=float,
        )

        # Clean physical state 000000000 is state 0.
        e0[0] = 1.0

        mean_cycle_lifetime = float(
            e0
            @ np.linalg.solve(
                np.eye(
                    K.shape[0]
                )
                - K,
                np.ones(
                    K.shape[0]
                ),
            )
        )

    except np.linalg.LinAlgError:

        pass

    return {
        "rho": rho,
        "h_cycle": h_cycle,
        "h_step": h_step,
        "mean_cycles": mean_cycle_lifetime,
    }


# =============================================================================
# Main
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--p",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--q",
        type=float,
        default=None,
        help="Default: q=p.",
    )

    parser.add_argument(
        "--shots-per-state",
        type=int,
        default=20_000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/pheno/harrington1d_cycle_markov.npz"
        ),
    )

    args = parser.parse_args()

    q = (
        args.p
        if args.q is None
        else args.q
    )

    print()
    print("=" * 88)
    print(
        "HARRINGTON1D d=9 CYCLE-BOUNDARY MARKOV MODEL"
    )
    print("=" * 88)

    print(
        f"p = {args.p}"
    )

    print(
        f"q = {q}"
    )

    print(
        f"shots/state = {args.shots_per_state}"
    )

    states, K, p_absorb, tau = (
        build_matrix(
            p=args.p,
            q=q,
            shots_per_state=args.shots_per_state,
            seed=args.seed,
        )
    )

    result = analyse_matrix(
        K,
        tau,
    )

    row_error = np.max(
        np.abs(
            K.sum(axis=1)
            + p_absorb
            - 1.0
        )
    )

    print()
    print("=" * 88)

    print(
        f"tau                  = {tau}"
    )

    print(
        f"rho(K_tau)           = {result['rho']:.10f}"
    )

    print(
        f"1-rho(K_tau)         = {result['h_cycle']:.6e}"
    )

    print(
        f"hazard / CA step     = {result['h_step']:.6e}"
    )

    if result[
        "mean_cycles"
    ] is not None:

        print(
            f"<T> from clean [cycles] = "
            f"{result['mean_cycles']:.6e}"
        )

        print(
            f"<T> from clean [steps]  = "
            f"{tau * result['mean_cycles']:.6e}"
        )

    print(
        f"max row normalization error = {row_error:.3e}"
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        args.output,
        states=np.asarray(
            states,
            dtype=np.int64,
        ),
        K=K,
        p_absorb=p_absorb,
        tau=tau,
        p=args.p,
        q=q,
        shots_per_state=args.shots_per_state,
    )

    print(
        f"saved: {args.output}"
    )


if __name__ == "__main__":
    main()
