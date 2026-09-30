"""
Optimized Harrington1D decoder.

Each uint64 packs 64 independent trajectories.

The implementation preserves explicit CountSig and FlipSig fields
at every physical site and hierarchy level. This makes the dynamics
directly compatible with later signal-noise simulations.

For p_signal = 0 the dynamics reproduce src/harrington1d.py.
"""

import math

import numpy as np
from numba import njit


LEFT = 0
RIGHT = 1
CENTER = 2
NONE = -1

MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


# ============================================================
# RNG
# ============================================================

@njit(inline="always")
def _rotl(x, k):
    return (
        (x << np.uint64(k))
        | (x >> np.uint64(64 - k))
    )


@njit(inline="always")
def _splitmix64(x):
    x += np.uint64(0x9E3779B97F4A7C15)

    z = x
    z = (
        (z ^ (z >> np.uint64(30)))
        * np.uint64(0xBF58476D1CE4E5B9)
    )
    z = (
        (z ^ (z >> np.uint64(27)))
        * np.uint64(0x94D049BB133111EB)
    )
    z ^= z >> np.uint64(31)

    return x, z


@njit
def _seed_rng(seed):
    state = np.empty(4, dtype=np.uint64)

    x = np.uint64(seed)

    for i in range(4):
        x, state[i] = _splitmix64(x)

    return state


@njit(inline="always")
def _rand_u64(state):
    result = (
        _rotl(
            state[1] * np.uint64(5),
            7,
        )
        * np.uint64(9)
    )

    t = state[1] << np.uint64(17)

    state[2] ^= state[0]
    state[3] ^= state[1]
    state[1] ^= state[2]
    state[0] ^= state[3]

    state[2] ^= t
    state[3] = _rotl(state[3], 45)

    return result


@njit(inline="always")
def _bernoulli_word(state, threshold):
    if threshold == 0:
        return np.uint64(0)

    if threshold >= 0x100000000:
        return MASK64

    equal = MASK64
    less = np.uint64(0)

    for bit in range(31, -1, -1):
        rnd = _rand_u64(state)

        if (threshold >> bit) & 1:
            less |= equal & ~rnd
            equal &= rnd
        else:
            equal &= ~rnd

    return less


# ============================================================
# Popcount
# ============================================================

@njit(inline="always")
def _popcount(x):
    x -= (
        (x >> np.uint64(1))
        & np.uint64(0x5555555555555555)
    )

    x = (
        x
        & np.uint64(0x3333333333333333)
    ) + (
        (x >> np.uint64(2))
        & np.uint64(0x3333333333333333)
    )

    x = (
        x
        + (x >> np.uint64(4))
    ) & np.uint64(0x0F0F0F0F0F0F0F0F)

    x += x >> np.uint64(8)
    x += x >> np.uint64(16)
    x += x >> np.uint64(32)

    return int(
        x & np.uint64(0x7F)
    )


# ============================================================
# Bit-sliced counters
# ============================================================

@njit(inline="always")
def _counter_increment(
    counters,
    counter_type,
    rep,
    nbits,
    mask,
):
    carry = mask

    for bit in range(nbits):
        old = counters[
            counter_type,
            bit,
            rep,
        ]

        counters[
            counter_type,
            bit,
            rep,
        ] = old ^ carry

        carry &= old


@njit(inline="always")
def _counter_geq(
    counters,
    counter_type,
    rep,
    nbits,
    threshold,
    valid_mask,
):
    greater = np.uint64(0)
    equal = valid_mask

    for bit in range(
        nbits - 1,
        -1,
        -1,
    ):
        value = counters[
            counter_type,
            bit,
            rep,
        ]

        if (threshold >> bit) & 1:
            equal &= value
        else:
            greater |= equal & value
            equal &= ~value

    return (
        greater | equal
    ) & valid_mask


@njit(inline="always")
def _counter_reset_lane(
    counters,
    rep,
    nbits,
    reset_mask,
):
    keep = ~reset_mask

    for counter_type in range(3):
        for bit in range(nbits):
            counters[
                counter_type,
                bit,
                rep,
            ] &= keep


# ============================================================
# Syndrome
# ============================================================

@njit
def _syndrome(
    qubits,
    defects,
    d,
):
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


@njit
def _syndrome_mask(
    qubits,
    defects,
    d,
):
    _syndrome(
        qubits,
        defects,
        d,
    )

    mask = np.uint64(0)

    for i in range(d):
        mask |= defects[i]

    return mask


# ============================================================
# Packed Harrington rule
# ============================================================

@njit(inline="always")
def _rule_masks(
    role,
    left,
    center,
    right,
    available,
):
    center &= available

    move_left = np.uint64(0)
    move_right = np.uint64(0)

    if role == LEFT:
        move_left = (
            center
            & left
        )

        move_right = (
            center
            & ~left
            & available
        )

    elif role == RIGHT:
        move_left = (
            center
            & ~right
            & available
        )

    return (
        move_left,
        move_right,
    )


# ============================================================
# One CA step
# ============================================================

@njit
def _step(
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

    valid_mask,
    signal_threshold,
    rng_state,

    d,
    n_levels,

    syndrome_precomputed=False,
):
    # --------------------------------------------------------
    # Syndrome
    # --------------------------------------------------------

    if not syndrome_precomputed:
        _syndrome(
            qubits,
            defects,
            d,
        )

    # --------------------------------------------------------
    # Update global level clocks
    # --------------------------------------------------------

    for level in range(n_levels):
        ages[level] = (
            ages[level] + 1
        ) % periods[level]

    # --------------------------------------------------------
    # Acquire + update signals and counters
    # --------------------------------------------------------

    for level in range(n_levels):

        for i in range(d):
            left = (
                i - 1
                if i > 0
                else d - 1
            )

            right = (
                i + 1
                if i + 1 < d
                else 0
            )

            rep = site_to_rep[
                level,
                i,
            ]

            incoming_count_L = (
                count_L[level, right]
            )

            incoming_count_R = (
                count_R[level, left]
            )

            incoming_flip_L = (
                flip_L[level, right]
            )

            incoming_flip_R = (
                flip_R[level, left]
            )

            if rep >= 0:
                # Representatives broadcast their own defect.
                new_count_L[
                    level,
                    i,
                ] = defects[i]

                new_count_R[
                    level,
                    i,
                ] = defects[i]

                # Representatives retain generated FlipSignals.
                new_flip_L[
                    level,
                    i,
                ] = flip_L[level, i]

                new_flip_R[
                    level,
                    i,
                ] = flip_R[level, i]

                nbits = nbits_levels[level]

                _counter_increment(
                    counters,
                    CENTER,
                    rep,
                    nbits,
                    defects[i],
                )

                _counter_increment(
                    counters,
                    LEFT,
                    rep,
                    nbits,
                    incoming_count_R,
                )

                _counter_increment(
                    counters,
                    RIGHT,
                    rep,
                    nbits,
                    incoming_count_L,
                )

            else:
                # Non-representatives simply relay signals.
                new_count_L[
                    level,
                    i,
                ] = incoming_count_L

                new_count_R[
                    level,
                    i,
                ] = incoming_count_R

                new_flip_L[
                    level,
                    i,
                ] = incoming_flip_L

                new_flip_R[
                    level,
                    i,
                ] = incoming_flip_R

    # Commit updated signal fields.
    for level in range(n_levels):
        for i in range(d):
            count_L[level, i] = (
                new_count_L[level, i]
            )

            count_R[level, i] = (
                new_count_R[level, i]
            )

            flip_L[level, i] = (
                new_flip_L[level, i]
            )

            flip_R[level, i] = (
                new_flip_R[level, i]
            )

    # --------------------------------------------------------
    # Rule: highest hierarchy level first
    # --------------------------------------------------------

    correction_L = np.zeros(
        d,
        dtype=np.uint64,
    )

    correction_R = np.zeros(
        d,
        dtype=np.uint64,
    )

    for i in range(d):

        available = valid_mask

        for level in range(
            n_levels - 1,
            -1,
            -1,
        ):
            rep = site_to_rep[
                level,
                i,
            ]

            age = ages[level]

            # -----------------------------------------------
            # Threshold accumulated counts
            # -----------------------------------------------

            if (
                rep >= 0
                and age
                == U_levels[level] - 1
            ):
                nbits = (
                    nbits_levels[level]
                )

                coarse_C = _counter_geq(
                    counters,
                    CENTER,
                    rep,
                    nbits,
                    threshold_C[level],
                    valid_mask,
                )

                coarse_L = _counter_geq(
                    counters,
                    LEFT,
                    rep,
                    nbits,
                    threshold_N[level],
                    valid_mask,
                )

                coarse_R = _counter_geq(
                    counters,
                    RIGHT,
                    rep,
                    nbits,
                    threshold_N[level],
                    valid_mask,
                )

                (
                    move_L,
                    move_R,
                ) = _rule_masks(
                    roles[level, i],
                    coarse_L,
                    coarse_C,
                    coarse_R,
                    available,
                )

                flip_L[
                    level,
                    i,
                ] |= move_L

                flip_R[
                    level,
                    i,
                ] |= move_R

            # -----------------------------------------------
            # Release FlipSignals and reset this level
            # -----------------------------------------------

            if (
                age
                == U_levels[level]
                + Q_levels[level]
                - 1
            ):
                move_L = (
                    flip_L[level, i]
                    & available
                )

                move_R = (
                    flip_R[level, i]
                    & available
                    & ~move_L
                )

                correction_L[i] |= move_L
                correction_R[i] |= move_R

                assigned = (
                    move_L | move_R
                )

                # Reset only trajectories that reached this
                # level in the top-down rule.
                reset_mask = available
                keep = ~reset_mask

                count_L[
                    level,
                    i,
                ] &= keep

                count_R[
                    level,
                    i,
                ] &= keep

                flip_L[
                    level,
                    i,
                ] &= keep

                flip_R[
                    level,
                    i,
                ] &= keep

                if rep >= 0:
                    _counter_reset_lane(
                        counters,
                        rep,
                        nbits_levels[level],
                        reset_mask,
                    )

                available &= ~assigned

        # ----------------------------------------------------
        # Physical fallback rule
        # ----------------------------------------------------

        if available != 0:
            left = (
                i - 1
                if i > 0
                else d - 1
            )

            right = (
                i + 1
                if i + 1 < d
                else 0
            )

            coordinate = i % 3

            if coordinate == 0:
                role = LEFT
            elif coordinate == 1:
                role = CENTER
            else:
                role = RIGHT

            (
                local_L,
                local_R,
            ) = _rule_masks(
                role,
                defects[left],
                defects[i],
                defects[right],
                available,
            )

            correction_L[i] |= (
                local_L
            )

            correction_R[i] |= (
                local_R
            )

    # --------------------------------------------------------
    # Apply all corrections modulo 2
    # --------------------------------------------------------

    flip_qubit = np.zeros(
        d,
        dtype=np.uint64,
    )

    for i in range(d):
        right = (
            i + 1
            if i + 1 < d
            else 0
        )

        flip_qubit[i] ^= (
            correction_L[i]
        )

        flip_qubit[right] ^= (
            correction_R[i]
        )

    for i in range(d):
        qubits[i] ^= flip_qubit[i]

    # --------------------------------------------------------
    # Optional signal-noise hook
    #
    # Noise is applied to stored signals at the end of the CA
    # step. Code-capacity simulations use signal_threshold=0.
    # --------------------------------------------------------

    if signal_threshold != 0:

        for level in range(n_levels):
            for i in range(d):

                count_L[
                    level,
                    i,
                ] ^= (
                    _bernoulli_word(
                        rng_state,
                        signal_threshold,
                    )
                    & valid_mask
                )

                count_R[
                    level,
                    i,
                ] ^= (
                    _bernoulli_word(
                        rng_state,
                        signal_threshold,
                    )
                    & valid_mask
                )

                flip_L[
                    level,
                    i,
                ] ^= (
                    _bernoulli_word(
                        rng_state,
                        signal_threshold,
                    )
                    & valid_mask
                )

                flip_R[
                    level,
                    i,
                ] ^= (
                    _bernoulli_word(
                        rng_state,
                        signal_threshold,
                    )
                    & valid_mask
                )


# ============================================================
# Hierarchy geometry
# ============================================================

def _hierarchy_depth(d):
    depth = 0
    size = 1

    while size < d:
        size *= 3
        depth += 1

    if size != d:
        raise ValueError(
            "Harrington1D requires d to be a power of 3."
        )

    return depth


def _role_from_coordinate(x):
    if x == 0:
        return LEFT

    if x == 1:
        return CENTER

    return RIGHT


def _build_geometry(
    d,
    U,
    fN,
    fC,
):
    depth = _hierarchy_depth(d)
    n_levels = depth - 1

    site_to_rep = np.full(
        (n_levels, d),
        -1,
        dtype=np.int64,
    )

    roles = np.full(
        (n_levels, d),
        NONE,
        dtype=np.int64,
    )

    U_levels = np.empty(
        n_levels,
        dtype=np.int64,
    )

    Q_levels = np.empty(
        n_levels,
        dtype=np.int64,
    )

    periods = np.empty(
        n_levels,
        dtype=np.int64,
    )

    nbits_levels = np.empty(
        n_levels,
        dtype=np.int64,
    )

    threshold_N = np.empty(
        n_levels,
        dtype=np.int64,
    )

    threshold_C = np.empty(
        n_levels,
        dtype=np.int64,
    )

    rep_count = 0

    for level in range(n_levels):
        k = level + 1

        Uk = U**k
        Qk = 3**k
        period = Uk + Qk

        U_levels[level] = Uk
        Q_levels[level] = Qk
        periods[level] = period

        nbits_levels[level] = max(
            1,
            int(
                math.ceil(
                    math.log2(
                        period + 1
                    )
                )
            ),
        )

        threshold_N[level] = int(
            math.ceil(fN * Uk)
        )

        threshold_C[level] = int(
            math.ceil(fC * Uk)
        )

        offset = (
            Qk - 1
        ) // 2

        for site in range(d):
            delta = site - offset

            if delta % Qk != 0:
                continue

            coarse_site = (
                delta // Qk
            )

            role = _role_from_coordinate(
                coarse_site % 3
            )

            site_to_rep[
                level,
                site,
            ] = rep_count

            roles[
                level,
                site,
            ] = role

            rep_count += 1

    max_bits = (
        int(nbits_levels.max())
        if n_levels
        else 1
    )

    return {
        "n_levels": n_levels,
        "site_to_rep": site_to_rep,
        "roles": roles,
        "U_levels": U_levels,
        "Q_levels": Q_levels,
        "periods": periods,
        "nbits_levels": nbits_levels,
        "threshold_N": threshold_N,
        "threshold_C": threshold_C,
        "rep_count": rep_count,
        "max_bits": max_bits,
    }


# ============================================================
# Packed simulation
# ============================================================

@njit
def _simulate_group(
    d,
    max_steps,
    p_threshold,
    signal_threshold,
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

    for i in range(d):
        qubits[i] = (
            _bernoulli_word(
                rng_state,
                p_threshold,
            )
            & valid_mask
        )

    count_L = np.zeros(
        (n_levels, d),
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

    ages = np.zeros(
        n_levels,
        dtype=np.int64,
    )

    alive = valid_mask
    logical_failures = 0

    for _ in range(max_steps + 1):

        syndrome = (
            _syndrome_mask(
                qubits,
                defects,
                d,
            )
            & alive
        )

        decoded = (
            alive
            & ~syndrome
        )

        if decoded != 0:
            logical = np.uint64(0)

            # With zero syndrome all qubits are equal,
            # so one physical qubit gives the logical sector.
            logical = (
                qubits[0]
                & decoded
            )

            logical_failures += (
                _popcount(logical)
            )

            alive &= ~decoded

        if alive == 0:
            break

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

            valid_mask,
            signal_threshold,
            rng_state,

            d,
            n_levels,
        )

    unresolved = _popcount(
        alive
    )

    return (
        logical_failures,
        unresolved,
    )


# ============================================================
# Public code-capacity interface
# ============================================================

def simulate_code_capacity(
    d,
    p,
    shots,
    U=10,
    fN=0.4,
    fC=0.9,
    max_steps=10_000,
    seed=1,
    p_signal=0.0,
):
    """
    Simulate Harrington1D.

    For code capacity use:

        p_signal = 0.0

    The signal-noise argument is already part of the kernel so
    later noisy-signal simulations use the same implementation.
    """

    d = int(d)
    shots = int(shots)
    U = int(U)
    max_steps = int(max_steps)
    seed = int(seed)

    if shots <= 0:
        raise ValueError(
            "shots must be positive"
        )

    if not 0.0 <= p <= 1.0:
        raise ValueError(
            "p must lie in [0,1]"
        )

    if not 0.0 <= p_signal <= 1.0:
        raise ValueError(
            "p_signal must lie in [0,1]"
        )

    geometry = _build_geometry(
        d,
        U,
        fN,
        fC,
    )

    def threshold(probability):
        if probability <= 0.0:
            return 0

        if probability >= 1.0:
            return 0x100000000

        return int(
            math.floor(
                probability
                * 4294967296.0
            )
        )

    p_threshold = threshold(p)

    signal_threshold = threshold(
        p_signal
    )

    rng_state = _seed_rng(
        seed
    )

    logical_failures = 0
    unresolved = 0

    remaining = shots

    while remaining > 0:

        group_size = min(
            64,
            remaining,
        )

        if group_size == 64:
            valid_mask = MASK64

        else:
            valid_mask = np.uint64(
                (1 << group_size) - 1
            )

        (
            group_logical,
            group_unresolved,
        ) = _simulate_group(
            d,
            max_steps,
            p_threshold,
            signal_threshold,
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

        logical_failures += int(
            group_logical
        )

        unresolved += int(
            group_unresolved
        )

        remaining -= group_size

    failures = (
        logical_failures
        + unresolved
    )

    return {
        "shots": shots,
        "failures": failures,
        "logical_failures": logical_failures,
        "unresolved": unresolved,
        "pL": failures / shots,
        "pL_logical": logical_failures / shots,
    }
