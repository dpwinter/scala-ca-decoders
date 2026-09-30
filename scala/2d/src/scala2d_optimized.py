"""
Fast multispin implementation of the SCALA2D paper decoder.

This module contains ONLY the SCALA dynamics.

It reproduces the paper-reference implementation in src/scala2d.py,
but packs 64 independent Monte Carlo trajectories into each uint64.

No MWPM logic lives here.  The caller receives the final residual
error configurations and can decide how to classify unresolved states.
"""

import math

import numpy as np
from numba import njit


MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


# =============================================================================
# RNG
# =============================================================================

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
    """
    Generate 64 Bernoulli(p) bits simultaneously.

    threshold = floor(p * 2^32).
    """
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


# =============================================================================
# Basic packed toric-code operations
# =============================================================================

@njit
def _syndrome(h, v, defect, d):
    for y in range(d):
        yp = y + 1 if y + 1 < d else 0

        for x in range(d):
            xp = x + 1 if x + 1 < d else 0

            defect[y, x] = (
                h[y, x]
                ^ h[y, xp]
                ^ v[y, x]
                ^ v[yp, x]
            )


@njit
def _clear_signals(N, W, E, S, d):
    for y in range(d):
        for x in range(d):
            N[y, x] = np.uint64(0)
            W[y, x] = np.uint64(0)
            E[y, x] = np.uint64(0)
            S[y, x] = np.uint64(0)


# =============================================================================
# One SCALA step
# =============================================================================

@njit
def _step(
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
    # -------------------------------------------------------------------------
    # Acquire
    # -------------------------------------------------------------------------

    _syndrome(
        h,
        v,
        defect,
        d,
    )

    # -------------------------------------------------------------------------
    # Broadcast + propagate
    #
    # This is algebraically equivalent to the explicit reference implementation:
    #
    #   a defect broadcasts along an axis iff it currently carries neither
    #   signal on that axis, then everything propagates by one lattice cell.
    # -------------------------------------------------------------------------

    for y in range(d):
        yp = y + 1 if y + 1 < d else 0
        ym = y - 1 if y > 0 else d - 1

        for x in range(d):
            xp = x + 1 if x + 1 < d else 0
            xm = x - 1 if x > 0 else d - 1

            # Incoming north-moving signal from southern cell.
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

            # Incoming south-moving signal from northern cell.
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

            # Incoming west-moving signal from eastern cell.
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

            # Incoming east-moving signal from western cell.
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
    #
    # Exact packed form of src.scala2d.correction().
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

            # Nearest-neighbor rule:
            # west has priority over north.
            move_w = D & DW
            move_n = D & ~DW & DN

            # Signal following only applies if the defect is isolated.
            isolated = (
                D
                & ~DN
                & ~DW
                & ~DE
                & ~DS
            )

            # Exact 16-entry signal-follow truth table.

            # W correction:
            # one E signal, NE, or NES.
            signal_w = (
                (e & ~n & ~w & ~s)
                | (n & e & ~w & ~s)
                | (n & e & s & ~w)
            )

            # E correction:
            # one W signal or NWS.
            signal_e = (
                (w & ~n & ~e & ~s)
                | (n & w & s & ~e)
            )

            # N correction:
            # one S signal or WES.
            signal_n = (
                (s & ~n & ~w & ~e)
                | (w & e & s & ~n)
            )

            # S correction:
            # one N signal, NW, or NWE.
            signal_s = (
                (n & ~w & ~e & ~s)
                | (n & w & ~e & ~s)
                | (n & w & e & ~s)
            )

            move_w |= isolated & signal_w
            move_e = isolated & signal_e
            move_n |= isolated & signal_n
            move_s = isolated & signal_s

            # Synchronous corrections are implemented by XORing every
            # requested edge flip into the residual data configuration.
            h[y, x] ^= move_w
            h[y, xp] ^= move_e

            v[y, x] ^= move_n
            v[yp, x] ^= move_s


# =============================================================================
# Full paper decoder
# =============================================================================

@njit
def _decode_paper(
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
    Exact paper reset schedule

        1,2,...,d-1,d,d-1,...,2,1

    with a complete signal reset after every block.

    Total runtime = d^2 CA steps.
    """

    # Ramp up.
    for reset_time in range(1, d + 1):

        for _ in range(reset_time):
            _step(
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
            )

        _clear_signals(
            N,
            W,
            E,
            S,
            d,
        )

    # Ramp down.
    for reset_time in range(d - 1, 0, -1):

        for _ in range(reset_time):
            _step(
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
            )

        _clear_signals(
            N,
            W,
            E,
            S,
            d,
        )


# =============================================================================
# Final masks
# =============================================================================

@njit
def _syndrome_mask(
    h,
    v,
    defect,
    d,
):
    _syndrome(
        h,
        v,
        defect,
        d,
    )

    mask = np.uint64(0)

    for y in range(d):
        for x in range(d):
            mask |= defect[y, x]

    return mask


@njit
def _logical_mask(
    h,
    v,
    d,
):
    """
    Raw toric homology mask.

    This is meaningful as a final logical classification only when
    syndrome == 0.

    For unresolved configurations, the run script should first complete
    the residual syndrome, e.g. with MWPM.
    """
    logical_x = np.uint64(0)
    logical_y = np.uint64(0)

    for x in range(d):
        logical_x ^= v[0, x]

    for y in range(d):
        logical_y ^= h[y, 0]

    return logical_x | logical_y


# =============================================================================
# One packed Monte Carlo group
# =============================================================================

@njit
def _simulate_group(
    d,
    threshold,
    valid_mask,
    rng_state,
):
    shape = (d, d)

    h = np.zeros(
        shape,
        dtype=np.uint64,
    )

    v = np.zeros(
        shape,
        dtype=np.uint64,
    )

    N = np.zeros(
        shape,
        dtype=np.uint64,
    )

    W = np.zeros(
        shape,
        dtype=np.uint64,
    )

    E = np.zeros(
        shape,
        dtype=np.uint64,
    )

    S = np.zeros(
        shape,
        dtype=np.uint64,
    )

    defect = np.zeros(
        shape,
        dtype=np.uint64,
    )

    new_N = np.zeros(
        shape,
        dtype=np.uint64,
    )

    new_W = np.zeros(
        shape,
        dtype=np.uint64,
    )

    new_E = np.zeros(
        shape,
        dtype=np.uint64,
    )

    new_S = np.zeros(
        shape,
        dtype=np.uint64,
    )

    # Initial independent code-capacity noise.
    for y in range(d):
        for x in range(d):

            h[y, x] = (
                _bernoulli_word(
                    rng_state,
                    threshold,
                )
                & valid_mask
            )

            v[y, x] = (
                _bernoulli_word(
                    rng_state,
                    threshold,
                )
                & valid_mask
            )

    # SCALA paper decoder.
    _decode_paper(
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
    )

    syn = (
        _syndrome_mask(
            h,
            v,
            defect,
            d,
        )
        & valid_mask
    )

    logical = (
        _logical_mask(
            h,
            v,
            d,
        )
        & valid_mask
    )

    return (
        h,
        v,
        syn,
        logical,
    )


# =============================================================================
# Public helpers
# =============================================================================

def paper_reset_schedule(d):
    """
    Explicit paper schedule, useful for diagnostics/tests.
    """
    d = int(d)

    return (
        list(range(1, d + 1))
        + list(range(d - 1, 0, -1))
    )


def extract_lane(packed, lane):
    """
    Extract one d x d trajectory from a packed uint64 lattice.

    Returned dtype is uint8 with entries 0/1.
    """
    return (
        (
            packed
            >> np.uint64(int(lane))
        )
        & np.uint64(1)
    ).astype(
        np.uint8,
        copy=False,
    )


def syndrome_array(h, v):
    """
    Ordinary NumPy syndrome for one explicit configuration.

    Same convention as src.scala2d.syndrome().
    """
    h = np.asarray(
        h,
        dtype=np.uint8,
    )

    v = np.asarray(
        v,
        dtype=np.uint8,
    )

    return (
        h
        ^ np.roll(
            h,
            -1,
            axis=1,
        )
        ^ v
        ^ np.roll(
            v,
            -1,
            axis=0,
        )
    ).astype(
        np.uint8,
        copy=False,
    )


def logical_error(h, v):
    """
    Homology of a syndrome-free explicit configuration.
    """
    h = np.asarray(
        h,
        dtype=np.uint8,
    )

    v = np.asarray(
        v,
        dtype=np.uint8,
    )

    lx = int(
        np.bitwise_xor.reduce(
            v[0, :]
        )
    )

    ly = int(
        np.bitwise_xor.reduce(
            h[:, 0]
        )
    )

    return bool(
        lx | ly
    )


# =============================================================================
# Public Monte Carlo generator
# =============================================================================

def simulate_code_capacity_residuals(
    d,
    p,
    shots,
    seed=1,
):
    """
    Run SCALA2D code-capacity Monte Carlo and return the FINAL residual
    configurations.

    No logical-failure convention is imposed here.

    Parameters
    ----------
    d : int
        Toric-code linear size.

    p : float
        Independent physical X-error probability per data qubit.

    shots : int
        Number of trajectories.

    seed : int
        RNG seed.

    Returns
    -------
    groups : list of dict
        One item per packed group of at most 64 trajectories.

        Each item contains

            h
                packed final horizontal residual configuration

            v
                packed final vertical residual configuration

            syndrome_mask
                packed mask selecting trajectories whose final syndrome
                is nonzero

            logical_mask
                raw homology mask of the final residual configuration

            valid_mask
                packed mask selecting real trajectories in the group

            group_size
                number of real trajectories in the group

    Notes
    -----
    For syndrome-free trajectories, logical_mask gives the final SCALA
    logical outcome directly.

    For trajectories selected by syndrome_mask, extract the corresponding
    h/v lane and perform the desired terminal classification externally,
    e.g. with PyMatching.
    """
    d = int(d)
    shots = int(shots)
    seed = int(seed)

    if d <= 1:
        raise ValueError(
            "d must be at least 2"
        )

    if shots <= 0:
        raise ValueError(
            "shots must be positive"
        )

    if not 0.0 <= p <= 1.0:
        raise ValueError(
            "p must lie in [0,1]"
        )

    if p <= 0.0:
        threshold = 0

    elif p >= 1.0:
        threshold = 0x100000000

    else:
        threshold = int(
            math.floor(
                p * 4294967296.0
            )
        )

    rng_state = _seed_rng(
        seed
    )

    groups = []

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

        h, v, syn, logical = (
            _simulate_group(
                d,
                threshold,
                valid_mask,
                rng_state,
            )
        )

        groups.append(
            {
                "h": h,
                "v": v,
                "syndrome_mask": (
                    syn & valid_mask
                ),
                "logical_mask": (
                    logical & valid_mask
                ),
                "valid_mask": valid_mask,
                "group_size": group_size,
            }
        )

        remaining -= group_size

    return groups
