"""
High-performance implementation of the paper SCALA1D decoder.

Each uint64 stores 64 independent Monte Carlo trajectories.

The local CA rule is exactly the same as in src/scala1d.py:

    1. Acquire measured syndrome
    2. Broadcast
    3. Propagate
    4. Correct

Only the implementation is optimized.
"""

import numpy as np
from numba import njit, prange


ALL = np.uint64(0xFFFFFFFFFFFFFFFF)

RNG_BITS = 24
RNG_SCALE = 1 << RNG_BITS


# ============================================================
# Random numbers
# ============================================================

@njit(inline="always")
def rng_next(state):
    """SplitMix64."""
    state += np.uint64(0x9E3779B97F4A7C15)

    z = state
    z = (
        (z ^ (z >> np.uint64(30)))
        * np.uint64(0xBF58476D1CE4E5B9)
    )
    z = (
        (z ^ (z >> np.uint64(27)))
        * np.uint64(0x94D049BB133111EB)
    )
    z ^= z >> np.uint64(31)

    return state, z


@njit(inline="always")
def bernoulli64(state, threshold):
    """Generate 64 independent Bernoulli bits."""

    if threshold <= 0:
        return state, np.uint64(0)

    if threshold >= RNG_SCALE:
        return state, ALL

    lt = np.uint64(0)
    eq = ALL

    for k in range(RNG_BITS - 1, -1, -1):
        state, r = rng_next(state)

        if (threshold >> k) & 1:
            lt |= eq & ~r
            eq &= r
        else:
            eq &= ~r

    return state, lt


# ============================================================
# SCALA1D local rule
# ============================================================

@njit(inline="always")
def scala_step(errors, left, right, defect, new_left, new_right, d):
    """
    One packed synchronous SCALA1D step.

    `defect` is the measured syndrome supplied to the decoder.
    """

    # --------------------------------------------------------
    # 1. Acquire
    #
    # `defect` already contains the measured syndrome.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # 2 + 3. Broadcast and propagate
    #
    # Cell j broadcasts iff
    #
    #     defect[j] = 1
    #     left[j]   = 0
    #     right[j]  = 0.
    #
    # The emitted signal then propagates one cell.
    # --------------------------------------------------------

    for i in range(d):

        ip = i + 1
        if ip == d:
            ip = 0

        im = i - 1
        if im < 0:
            im = d - 1

        emit_left = (
            defect[ip]
            & ~left[ip]
            & ~right[ip]
        )

        emit_right = (
            defect[im]
            & ~left[im]
            & ~right[im]
        )

        new_left[i] = (
            left[ip] | emit_left
        )

        new_right[i] = (
            right[im] | emit_right
        )

    # --------------------------------------------------------
    # 4. Correct
    #
    # Work directly qubit-by-qubit.
    #
    # Qubit i lies between cells i and i+1.
    # It can be flipped by exactly one of:
    #
    #   (a) nearest-neighbor annihilation,
    #   (b) cell i following a left-moving signal,
    #   (c) cell i+1 following a right-moving signal.
    #
    # These three conditions are mutually exclusive.
    # --------------------------------------------------------

    for i in range(d):

        im = i - 1
        if im < 0:
            im = d - 1

        ip = i + 1
        if ip == d:
            ip = 0

        ipp = ip + 1
        if ipp == d:
            ipp = 0

        # Neighboring defects across qubit i.
        nearest = (
            defect[i]
            & defect[ip]
        )

        # Cell i is isolated and moves right.
        follow_from_left_cell = (
            ~defect[im]
            & defect[i]
            & ~defect[ip]
            & new_left[i]
            & ~new_right[i]
        )

        # Cell i+1 is isolated and moves left.
        follow_from_right_cell = (
            ~defect[i]
            & defect[ip]
            & ~defect[ipp]
            & new_right[ip]
            & ~new_left[ip]
        )

        correction = (
            nearest
            | follow_from_left_cell
            | follow_from_right_cell
        )

        errors[i] ^= correction

    # Store propagated signals.
    for i in range(d):
        left[i] = new_left[i]
        right[i] = new_right[i]


# ============================================================
# Logical sector
# ============================================================

@njit
def logical_mask(errors, d, valid):
    """
    Return one packed bit per trajectory indicating

        weight(errors) >= (d+1)/2.

    For odd d this is equivalent to weight(errors) > d//2.
    """

    nbits = 1
    x = d

    while x > 1:
        nbits += 1
        x >>= 1

    count = np.zeros(nbits, dtype=np.uint64)

    # Bit-sliced Hamming-weight counter.
    for i in range(d):
        carry = errors[i]
        k = 0

        while carry != 0 and k < nbits:
            old = count[k]
            count[k] = old ^ carry
            carry &= old
            k += 1

    threshold = d // 2 + 1

    gt = np.uint64(0)
    eq = ALL

    for k in range(nbits - 1, -1, -1):

        bit = count[k]

        if (threshold >> k) & 1:
            eq &= bit
        else:
            gt |= eq & bit
            eq &= ~bit

    return (gt | eq) & valid


# ============================================================
# Phenomenological noise
# ============================================================

@njit(parallel=True)
def _pheno_kernel(
    d,
    threshold_data,
    threshold_meas,
    shots,
    rounds,
    reset_period,
    seed,
):
    """
    Return first logical-failure round for every trajectory.

    -1 means no failure occurred within `rounds`.
    """

    first_failure = np.full(
        shots,
        -1,
        dtype=np.int32,
    )

    batches = (shots + 63) // 64

    for batch in prange(batches):

        start = batch * 64
        n = min(64, shots - start)

        if n == 64:
            valid = ALL
        else:
            valid = (
                np.uint64(1) << np.uint64(n)
            ) - np.uint64(1)

        alive = valid

        rng = (
            np.uint64(seed)
            + np.uint64(batch + 1)
            * np.uint64(0xD1B54A32D192ED03)
        )

        errors = np.zeros(d, dtype=np.uint64)
        left = np.zeros(d, dtype=np.uint64)
        right = np.zeros(d, dtype=np.uint64)

        defect = np.empty(d, dtype=np.uint64)
        new_left = np.empty(d, dtype=np.uint64)
        new_right = np.empty(d, dtype=np.uint64)

        for t in range(1, rounds + 1):

            if alive == 0:
                break

            # --------------------------------------------
            # Data noise
            # --------------------------------------------

            for i in range(d):

                rng, noise = bernoulli64(
                    rng,
                    threshold_data,
                )

                errors[i] ^= noise & alive

            # --------------------------------------------
            # Measured syndrome
            # --------------------------------------------

            for i in range(d):

                im = i - 1
                if im < 0:
                    im = d - 1

                rng, measurement_error = bernoulli64(
                    rng,
                    threshold_meas,
                )

                defect[i] = (
                    errors[im]
                    ^ errors[i]
                    ^ (measurement_error & alive)
                )

            # --------------------------------------------
            # SCALA
            # --------------------------------------------

            scala_step(
                errors,
                left,
                right,
                defect,
                new_left,
                new_right,
                d,
            )

            # --------------------------------------------
            # Periodic signal reset
            # --------------------------------------------

            if (
                reset_period > 0
                and t % reset_period == 0
            ):
                for i in range(d):
                    left[i] = np.uint64(0)
                    right[i] = np.uint64(0)

            # --------------------------------------------
            # First logical passage
            # --------------------------------------------

            failed = logical_mask(
                errors,
                d,
                alive,
            )

            if failed != 0:

                for lane in range(n):

                    bit = (
                        np.uint64(1)
                        << np.uint64(lane)
                    )

                    if failed & bit:
                        first_failure[start + lane] = t

                alive &= ~failed

    return first_failure


def pheno(
    d,
    p,
    shots,
    rounds,
    reset_period,
    p_meas=None,
    seed=1,
):
    """
    Phenomenological first-passage simulation.

    Parameters
    ----------
    d : int
        Odd code distance.

    p : float
        Data-qubit error probability per round.

    shots : int
        Number of trajectories.

    rounds : int
        Maximum number of CA rounds.

    reset_period : int or None
        Signal reset period.
        None or 0 means no resets.

    p_meas : float or None
        Measurement error probability.
        Defaults to p.

    seed : int
        RNG seed.

    Returns
    -------
    ndarray
        First logical-failure round for each trajectory.
        -1 means survival through all simulated rounds.
    """

    if d % 2 == 0:
        raise ValueError("SCALA1D simulations assume odd d.")

    if p_meas is None:
        p_meas = p

    if reset_period is None:
        reset_period = 0

    threshold_data = int(round(p * RNG_SCALE))
    threshold_meas = int(round(p_meas * RNG_SCALE))

    return _pheno_kernel(
        int(d),
        threshold_data,
        threshold_meas,
        int(shots),
        int(rounds),
        int(reset_period),
        int(seed),
    )


# ============================================================
# Code capacity
# ============================================================

@njit(parallel=True)
def _code_capacity_kernel(
    d,
    threshold,
    shots,
    rounds,
    seed,
):
    result = np.zeros(
        shots,
        dtype=np.uint8,
    )

    batches = (shots + 63) // 64

    for batch in prange(batches):

        start = batch * 64
        n = min(64, shots - start)

        if n == 64:
            valid = ALL
        else:
            valid = (
                np.uint64(1)
                << np.uint64(n)
            ) - np.uint64(1)

        rng = (
            np.uint64(seed)
            + np.uint64(batch + 1)
            * np.uint64(0xD1B54A32D192ED03)
        )

        errors = np.zeros(d, dtype=np.uint64)
        left = np.zeros(d, dtype=np.uint64)
        right = np.zeros(d, dtype=np.uint64)

        defect = np.empty(d, dtype=np.uint64)
        new_left = np.empty(d, dtype=np.uint64)
        new_right = np.empty(d, dtype=np.uint64)

        # Initial physical errors.
        for i in range(d):

            rng, noise = bernoulli64(
                rng,
                threshold,
            )

            errors[i] = noise & valid

        # Noise-free decoding.
        for _ in range(rounds):

            for i in range(d):

                im = i - 1
                if im < 0:
                    im = d - 1

                defect[i] = (
                    errors[im] ^ errors[i]
                )

            scala_step(
                errors,
                left,
                right,
                defect,
                new_left,
                new_right,
                d,
            )

        failed = logical_mask(
            errors,
            d,
            valid,
        )

        for lane in range(n):

            bit = (
                np.uint64(1)
                << np.uint64(lane)
            )

            if failed & bit:
                result[start + lane] = 1

    return result


def code_capacity_failures(
    d,
    p,
    shots,
    rounds=None,
    seed=1,
):
    """Return (number of logical failures, number of shots)."""

    if d % 2 == 0:
        raise ValueError("SCALA1D simulations assume odd d.")

    if rounds is None:
        rounds = max(1, d - 2)

    threshold = int(round(p * RNG_SCALE))

    result = _code_capacity_kernel(
        int(d),
        threshold,
        int(shots),
        int(rounds),
        int(seed),
    )

    return int(result.sum()), int(shots)


def code_capacity(
    d,
    p,
    shots,
    rounds=None,
    seed=1,
):
    """
    Estimate the code-capacity logical error probability.

    By default SCALA is run for d-2 rounds, the maximum
    erosion time stated for the paper decoder.
    """

    if d % 2 == 0:
        raise ValueError("SCALA1D simulations assume odd d.")

    if rounds is None:
        rounds = max(1, d - 2)

    threshold = int(round(p * RNG_SCALE))

    result = _code_capacity_kernel(
        int(d),
        threshold,
        int(shots),
        int(rounds),
        int(seed),
    )

    return result.mean()
