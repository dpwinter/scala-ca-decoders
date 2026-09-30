"""
Optimized Harrington2D implementation.

Each uint64 packs 64 independent trajectories.

The implementation preserves explicit per-site signals:
    8 CountSignals per hierarchy level
    4 FlipSignals per hierarchy level

so signal noise can later be added without changing the
decoder dynamics.

For p_count_signal = p_flip_signal = 0 this should reproduce src/harrington2d.py.
"""

import math

import numpy as np
from numba import njit


# ============================================================
# Locations
# ============================================================

N = 0
W = 1
E = 2
S = 3

NW = 4
NE = 5
SW = 6
SE = 7

C = 8
NONE = -1

N_DIR = 8
N_CARD = 4

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
    state = np.empty(
        4,
        dtype=np.uint64,
    )

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
    state[3] = _rotl(
        state[3],
        45,
    )

    return result


@njit(inline="always")
def _bernoulli_word(
    state,
    threshold,
):
    if threshold == 0:
        return np.uint64(0)

    if threshold >= 0x100000000:
        return MASK64

    equal = MASK64
    less = np.uint64(0)

    for bit in range(
        31,
        -1,
        -1,
    ):
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


@njit
def _reset_rep_counters(
    counters,
    rep,
    nbits,
):
    for counter_type in range(9):
        for bit in range(nbits):
            counters[
                counter_type,
                bit,
                rep,
            ] = np.uint64(0)


# ============================================================
# Toric syndrome
# ============================================================

@njit
def _syndrome(
    horizontal,
    vertical,
    defects,
    d,
):
    for y in range(d):
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

            defects[y, x] = (
                horizontal[y, x]
                ^ horizontal[y, xp]
                ^ vertical[ym, x]
                ^ vertical[y, x]
            )


@njit
def _syndrome_mask(
    horizontal,
    vertical,
    defects,
    d,
):
    _syndrome(
        horizontal,
        vertical,
        defects,
        d,
    )

    mask = np.uint64(0)

    for y in range(d):
        for x in range(d):
            mask |= defects[y, x]

    return mask


# ============================================================
# Logical sector
# ============================================================

@njit
def _logical_mask(
    horizontal,
    vertical,
    d,
):
    logical_h = np.uint64(0)
    logical_v = np.uint64(0)

    for y in range(d):
        logical_h ^= horizontal[y, 0]

    for x in range(d):
        logical_v ^= vertical[0, x]

    return logical_h | logical_v


# ============================================================
# Packed local Harrington rule
# ============================================================

@njit(inline="always")
def _rule_masks(
    role,
    defects,
    available,
):
    move_N = np.uint64(0)
    move_W = np.uint64(0)
    move_E = np.uint64(0)
    move_S = np.uint64(0)

    center = (
        defects[C]
        & available
    )

    if role == C or role == NONE:
        return (
            move_N,
            move_W,
            move_E,
            move_S,
        )

    if role == NW:
        rem = center

        rem &= ~defects[N]

        hit = rem & defects[W]
        move_W |= hit
        rem &= ~defects[W]

        hit = rem & defects[E]
        move_E |= hit
        rem &= ~defects[E]

        hit = rem & defects[S]
        move_S |= hit
        rem &= ~defects[S]

        hit = rem & defects[NW]
        move_W |= hit
        rem &= ~defects[NW]

        rem &= ~defects[NE]

        hit = rem & defects[SW]
        move_W |= hit
        rem &= ~defects[SW]

        move_E |= rem

    elif role == N:
        rem = center

        rem &= ~defects[N]
        rem &= ~defects[W]
        rem &= ~defects[E]

        hit = rem & defects[S]
        move_S |= hit
        rem &= ~defects[S]

        rem &= ~defects[NW]
        rem &= ~defects[NE]

        move_S |= rem

    elif role == NE:
        rem = center

        rem &= ~defects[N]
        rem &= ~defects[E]

        hit = rem & defects[W]
        move_W |= hit
        rem &= ~defects[W]

        hit = rem & defects[S]
        move_S |= hit
        rem &= ~defects[S]

        rem &= ~defects[NW]
        rem &= ~defects[NE]
        rem &= ~defects[SE]

        move_W |= rem

    elif role == W:
        rem = center

        hit = rem & defects[W]
        move_W |= hit
        rem &= ~defects[W]

        rem &= ~defects[N]
        rem &= ~defects[S]

        hit = rem & defects[E]
        move_E |= hit
        rem &= ~defects[E]

        hit = rem & defects[NW]
        move_W |= hit
        rem &= ~defects[NW]

        hit = rem & defects[SW]
        move_W |= hit
        rem &= ~defects[SW]

        move_E |= rem

    elif role == E:
        rem = center

        rem &= ~defects[E]
        rem &= ~defects[N]
        rem &= ~defects[S]

        hit = rem & defects[W]
        move_W |= hit
        rem &= ~defects[W]

        rem &= ~defects[NE]
        rem &= ~defects[SE]

        move_W |= rem

    elif role == SW:
        rem = center

        hit = rem & defects[W]
        move_W |= hit
        rem &= ~defects[W]

        hit = rem & defects[S]
        move_S |= hit
        rem &= ~defects[S]

        hit = rem & defects[N]
        move_N |= hit
        rem &= ~defects[N]

        hit = rem & defects[E]
        move_E |= hit
        rem &= ~defects[E]

        hit = rem & defects[NW]
        move_W |= hit
        rem &= ~defects[NW]

        hit = rem & defects[SW]
        move_S |= hit
        rem &= ~defects[SW]

        hit = rem & defects[SE]
        move_S |= hit
        rem &= ~defects[SE]

        move_E |= rem

    elif role == S:
        rem = center

        hit = rem & defects[S]
        move_S |= hit
        rem &= ~defects[S]

        rem &= ~defects[W]
        rem &= ~defects[E]

        hit = rem & defects[N]
        move_N |= hit
        rem &= ~defects[N]

        hit = rem & defects[SE]
        move_S |= hit
        rem &= ~defects[SE]

        hit = rem & defects[SW]
        move_S |= hit
        rem &= ~defects[SW]

        move_N |= rem

    elif role == SE:
        rem = center

        rem &= ~defects[E]

        hit = rem & defects[S]
        move_S |= hit
        rem &= ~defects[S]

        hit = rem & defects[W]
        move_W |= hit
        rem &= ~defects[W]

        hit = rem & defects[N]
        move_N |= hit
        rem &= ~defects[N]

        rem &= ~defects[NE]
        rem &= ~defects[SE]

        hit = rem & defects[SW]
        move_S |= hit
        rem &= ~defects[SW]

        move_W |= rem

    return (
        move_N,
        move_W,
        move_E,
        move_S,
    )


# ============================================================
# Local neighborhood
# ============================================================

@njit(inline="always")
def _local_defects(
    defects,
    y,
    x,
    d,
    out,
):
    ym = y - 1 if y > 0 else d - 1
    yp = y + 1 if y + 1 < d else 0

    xm = x - 1 if x > 0 else d - 1
    xp = x + 1 if x + 1 < d else 0

    out[N] = defects[ym, x]
    out[W] = defects[y, xm]
    out[E] = defects[y, xp]
    out[S] = defects[yp, x]

    out[NW] = defects[ym, xm]
    out[NE] = defects[ym, xp]
    out[SW] = defects[yp, xm]
    out[SE] = defects[yp, xp]

    out[C] = defects[y, x]


# ============================================================
# One complete CA step
# ============================================================

@njit
def _step(
    horizontal,
    vertical,
    defects,

    count_sig,
    flip_sig,

    new_count_sig,
    new_flip_sig,

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
    rng_state,

    d,
    n_levels,

    syndrome_precomputed=False,
):
    if not syndrome_precomputed:
        _syndrome(
            horizontal,
            vertical,
            defects,
            d,
        )

    # --------------------------------------------------------
    # Advance level clocks
    # --------------------------------------------------------

    for level in range(n_levels):
        ages[level] = (
            ages[level] + 1
        ) % periods[level]

    # --------------------------------------------------------
    # Acquire signals synchronously
    # --------------------------------------------------------

    for level in range(n_levels):

        for y in range(d):
            ym = y - 1 if y > 0 else d - 1
            yp = y + 1 if y + 1 < d else 0

            for x in range(d):
                xm = x - 1 if x > 0 else d - 1
                xp = x + 1 if x + 1 < d else 0

                new_count_sig[level, y, x, N] = (
                    count_sig[level, yp, x, N]
                )

                new_count_sig[level, y, x, W] = (
                    count_sig[level, y, xp, W]
                )

                new_count_sig[level, y, x, E] = (
                    count_sig[level, y, xm, E]
                )

                new_count_sig[level, y, x, S] = (
                    count_sig[level, ym, x, S]
                )

                new_count_sig[level, y, x, NW] = (
                    count_sig[level, yp, xp, NW]
                )

                new_count_sig[level, y, x, NE] = (
                    count_sig[level, yp, xm, NE]
                )

                new_count_sig[level, y, x, SW] = (
                    count_sig[level, ym, xp, SW]
                )

                new_count_sig[level, y, x, SE] = (
                    count_sig[level, ym, xm, SE]
                )

                new_flip_sig[level, y, x, N] = (
                    flip_sig[level, yp, x, N]
                )

                new_flip_sig[level, y, x, W] = (
                    flip_sig[level, y, xp, W]
                )

                new_flip_sig[level, y, x, E] = (
                    flip_sig[level, y, xm, E]
                )

                new_flip_sig[level, y, x, S] = (
                    flip_sig[level, ym, x, S]
                )

    # --------------------------------------------------------
    # Update hierarchy memory
    # --------------------------------------------------------

    for level in range(n_levels):

        for y in range(d):
            for x in range(d):

                rep = site_to_rep[
                    level,
                    y,
                    x,
                ]

                if rep >= 0:

                    for direction in range(N_DIR):
                        count_sig[
                            level,
                            y,
                            x,
                            direction,
                        ] = defects[y, x]

                    nbits = nbits_levels[level]

                    _counter_increment(
                        counters,
                        C,
                        rep,
                        nbits,
                        defects[y, x],
                    )

                    _counter_increment(
                        counters,
                        N,
                        rep,
                        nbits,
                        new_count_sig[
                            level,
                            y,
                            x,
                            S,
                        ],
                    )

                    _counter_increment(
                        counters,
                        W,
                        rep,
                        nbits,
                        new_count_sig[
                            level,
                            y,
                            x,
                            E,
                        ],
                    )

                    _counter_increment(
                        counters,
                        E,
                        rep,
                        nbits,
                        new_count_sig[
                            level,
                            y,
                            x,
                            W,
                        ],
                    )

                    _counter_increment(
                        counters,
                        S,
                        rep,
                        nbits,
                        new_count_sig[
                            level,
                            y,
                            x,
                            N,
                        ],
                    )

                    _counter_increment(
                        counters,
                        NW,
                        rep,
                        nbits,
                        new_count_sig[
                            level,
                            y,
                            x,
                            SE,
                        ],
                    )

                    _counter_increment(
                        counters,
                        NE,
                        rep,
                        nbits,
                        new_count_sig[
                            level,
                            y,
                            x,
                            SW,
                        ],
                    )

                    _counter_increment(
                        counters,
                        SW,
                        rep,
                        nbits,
                        new_count_sig[
                            level,
                            y,
                            x,
                            NE,
                        ],
                    )

                    _counter_increment(
                        counters,
                        SE,
                        rep,
                        nbits,
                        new_count_sig[
                            level,
                            y,
                            x,
                            NW,
                        ],
                    )

                else:

                    for direction in range(N_DIR):
                        count_sig[
                            level,
                            y,
                            x,
                            direction,
                        ] = new_count_sig[
                            level,
                            y,
                            x,
                            direction,
                        ]

                    for direction in range(N_CARD):
                        flip_sig[
                            level,
                            y,
                            x,
                            direction,
                        ] = new_flip_sig[
                            level,
                            y,
                            x,
                            direction,
                        ]

    # --------------------------------------------------------
    # Determine correction requests
    # --------------------------------------------------------

    correction_N = np.zeros(
        (d, d),
        dtype=np.uint64,
    )

    correction_W = np.zeros(
        (d, d),
        dtype=np.uint64,
    )

    correction_E = np.zeros(
        (d, d),
        dtype=np.uint64,
    )

    correction_S = np.zeros(
        (d, d),
        dtype=np.uint64,
    )

    coarse = np.zeros(
        9,
        dtype=np.uint64,
    )

    local = np.zeros(
        9,
        dtype=np.uint64,
    )

    for y in range(d):
        for x in range(d):

            available = valid_mask

            for level in range(
                n_levels - 1,
                -1,
                -1,
            ):
                rep = site_to_rep[
                    level,
                    y,
                    x,
                ]

                age = ages[level]

                # --------------------------------------------
                # Threshold
                # --------------------------------------------

                if (
                    rep >= 0
                    and age == U_levels[level]
                ):
                    nbits = nbits_levels[
                        level
                    ]

                    for loc in range(8):
                        coarse[loc] = _counter_geq(
                            counters,
                            loc,
                            rep,
                            nbits,
                            threshold_N[level],
                            valid_mask,
                        )

                    coarse[C] = _counter_geq(
                        counters,
                        C,
                        rep,
                        nbits,
                        threshold_C[level],
                        valid_mask,
                    )

                    (
                        move_N,
                        move_W,
                        move_E,
                        move_S,
                    ) = _rule_masks(
                        roles[level, y, x],
                        coarse,
                        valid_mask,
                    )

                    flip_sig[
                        level,
                        y,
                        x,
                        N,
                    ] |= move_N

                    flip_sig[
                        level,
                        y,
                        x,
                        W,
                    ] |= move_W

                    flip_sig[
                        level,
                        y,
                        x,
                        E,
                    ] |= move_E

                    flip_sig[
                        level,
                        y,
                        x,
                        S,
                    ] |= move_S

                    # Reference C++ clears count[] here.
                    _reset_rep_counters(
                        counters,
                        rep,
                        nbits,
                    )

                # --------------------------------------------
                # Release + exact Memory.reset()
                # --------------------------------------------

                if (
                    age
                    == U_levels[level]
                    + Q_levels[level]
                ):
                    move_N = (
                        flip_sig[
                            level,
                            y,
                            x,
                            N,
                        ]
                        & available
                    )

                    remaining = (
                        available
                        & ~move_N
                    )

                    move_W = (
                        flip_sig[
                            level,
                            y,
                            x,
                            W,
                        ]
                        & remaining
                    )

                    remaining &= ~move_W

                    move_E = (
                        flip_sig[
                            level,
                            y,
                            x,
                            E,
                        ]
                        & remaining
                    )

                    remaining &= ~move_E

                    move_S = (
                        flip_sig[
                            level,
                            y,
                            x,
                            S,
                        ]
                        & remaining
                    )

                    assigned = (
                        move_N
                        | move_W
                        | move_E
                        | move_S
                    )

                    correction_N[y, x] |= move_N
                    correction_W[y, x] |= move_W
                    correction_E[y, x] |= move_E
                    correction_S[y, x] |= move_S

                    available &= ~assigned

                    # Exact equivalent of Memory.reset().
                    for direction in range(N_DIR):
                        count_sig[
                            level,
                            y,
                            x,
                            direction,
                        ] = np.uint64(0)

                    for direction in range(N_CARD):
                        flip_sig[
                            level,
                            y,
                            x,
                            direction,
                        ] = np.uint64(0)

                    if rep >= 0:
                        _reset_rep_counters(
                            counters,
                            rep,
                            nbits_levels[level],
                        )

            # ------------------------------------------------
            # Physical fallback
            # ------------------------------------------------

            if available != 0:

                _local_defects(
                    defects,
                    y,
                    x,
                    d,
                    local,
                )

                row = y % 3
                col = x % 3

                if row == 0:
                    if col == 0:
                        role = NW
                    elif col == 1:
                        role = N
                    else:
                        role = NE

                elif row == 1:
                    if col == 0:
                        role = W
                    elif col == 1:
                        role = C
                    else:
                        role = E

                else:
                    if col == 0:
                        role = SW
                    elif col == 1:
                        role = S
                    else:
                        role = SE

                (
                    move_N,
                    move_W,
                    move_E,
                    move_S,
                ) = _rule_masks(
                    role,
                    local,
                    available,
                )

                correction_N[y, x] |= move_N
                correction_W[y, x] |= move_W
                correction_E[y, x] |= move_E
                correction_S[y, x] |= move_S

    # --------------------------------------------------------
    # Apply physical edge flips
    # --------------------------------------------------------

    flip_h = np.zeros(
        (d, d),
        dtype=np.uint64,
    )

    flip_v = np.zeros(
        (d, d),
        dtype=np.uint64,
    )

    for y in range(d):
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

            flip_v[ym, x] ^= (
                correction_N[y, x]
            )

            flip_h[y, x] ^= (
                correction_W[y, x]
            )

            flip_h[y, xp] ^= (
                correction_E[y, x]
            )

            flip_v[y, x] ^= (
                correction_S[y, x]
            )

    for y in range(d):
        for x in range(d):

            horizontal[y, x] ^= (
                flip_h[y, x]
            )

            vertical[y, x] ^= (
                flip_v[y, x]
            )


# ============================================================
# Geometry
# ============================================================

def _hierarchy_depth(d):
    depth = 0
    size = 1

    while size < d:
        size *= 3
        depth += 1

    if size != d:
        raise ValueError(
            "Harrington2D requires d to be a power of 3."
        )

    return depth


def _role(row, col):
    table = (
        (NW, N, NE),
        (W, C, E),
        (SW, S, SE),
    )

    return table[row][col]


def _build_geometry(
    d,
    U,
    fN,
    fC,
):
    depth = _hierarchy_depth(d)
    n_levels = depth - 1

    site_to_rep = np.full(
        (n_levels, d, d),
        -1,
        dtype=np.int64,
    )

    roles = np.full(
        (n_levels, d, d),
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

        U_levels[level] = Uk
        Q_levels[level] = Qk

        periods[level] = (
            Uk + Qk + 1
        )

        nbits_levels[level] = max(
            1,
            int(
                math.ceil(
                    math.log2(
                        Uk + Qk + 2
                    )
                )
            ),
        )

        threshold_N[level] = int(
            math.ceil(
                fN * Uk
            )
        )

        threshold_C[level] = int(
            math.ceil(
                fC * Uk
            )
        )

        offset = (
            Qk - 1
        ) // 2

        for y in range(d):
            dy = y - offset

            if dy % Qk != 0:
                continue

            coarse_y = (
                dy // Qk
            )

            for x in range(d):
                dx = x - offset

                if dx % Qk != 0:
                    continue

                coarse_x = (
                    dx // Qk
                )

                role = _role(
                    coarse_y % 3,
                    coarse_x % 3,
                )

                site_to_rep[
                    level,
                    y,
                    x,
                ] = rep_count

                roles[
                    level,
                    y,
                    x,
                ] = role

                rep_count += 1

    max_bits = (
        int(
            nbits_levels.max()
        )
        if n_levels > 0
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
# Monte Carlo group
# ============================================================
# Post-update signal noise
# ============================================================

@njit
def _apply_signal_noise(
    count_sig,
    flip_sig,
    count_signal_threshold,
    flip_signal_threshold,
    active_mask,
    rng_state,
    d,
    n_levels,
):
    """
    Apply independent bit-flip noise to the persistent Harrington
    signal memories after a complete CA update.

    CountSignal and FlipSignal bits have independent error rates.
    Only active packed trajectories are modified.
    """

    if count_signal_threshold != 0:
        for level in range(n_levels):
            for y in range(d):
                for x in range(d):
                    for direction in range(N_DIR):
                        count_sig[
                            level, y, x, direction
                        ] ^= (
                            _bernoulli_word(
                                rng_state,
                                count_signal_threshold,
                            )
                            & active_mask
                        )

    if flip_signal_threshold != 0:
        for level in range(n_levels):
            for y in range(d):
                for x in range(d):
                    for direction in range(N_CARD):
                        flip_sig[
                            level, y, x, direction
                        ] ^= (
                            _bernoulli_word(
                                rng_state,
                                flip_signal_threshold,
                            )
                            & active_mask
                        )


# ============================================================

@njit
def _simulate_group(
    d,
    max_steps,

    p_threshold,
    count_signal_threshold,
    flip_signal_threshold,

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
    horizontal = np.zeros(
        (d, d),
        dtype=np.uint64,
    )

    vertical = np.zeros(
        (d, d),
        dtype=np.uint64,
    )

    defects = np.zeros(
        (d, d),
        dtype=np.uint64,
    )

    for y in range(d):
        for x in range(d):

            horizontal[y, x] = (
                _bernoulli_word(
                    rng_state,
                    p_threshold,
                )
                & valid_mask
            )

            vertical[y, x] = (
                _bernoulli_word(
                    rng_state,
                    p_threshold,
                )
                & valid_mask
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

    new_count_sig = np.zeros_like(
        count_sig
    )

    new_flip_sig = np.zeros_like(
        flip_sig
    )

    counters = np.zeros(
        (
            9,
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

    for _ in range(
        max_steps + 1
    ):

        syndrome_mask = (
            _syndrome_mask(
                horizontal,
                vertical,
                defects,
                d,
            )
            & alive
        )

        resolved = (
            alive
            & ~syndrome_mask
        )

        if resolved != 0:

            logical = (
                _logical_mask(
                    horizontal,
                    vertical,
                    d,
                )
                & resolved
            )

            logical_failures += (
                _popcount(logical)
            )

            alive &= ~resolved

        if alive == 0:
            break

        _step(
            horizontal,
            vertical,
            defects,

            count_sig,
            flip_sig,

            new_count_sig,
            new_flip_sig,

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
            rng_state,

            d,
            n_levels,
        )

        _apply_signal_noise(
            count_sig,
            flip_sig,
            count_signal_threshold,
            flip_signal_threshold,
            alive,
            rng_state,
            d,
            n_levels,
        )

    unresolved = (
        _popcount(
            alive
        )
    )

    return (
        logical_failures,
        unresolved,
    )


# ============================================================
# Probability conversion
# ============================================================

def _probability_threshold(p):
    if p <= 0.0:
        return 0

    if p >= 1.0:
        return 0x100000000

    return int(
        math.floor(
            p * 4294967296.0
        )
    )


# ============================================================
# Public simulation interface
# ============================================================

def simulate_code_capacity(
    d,
    p,
    shots,
    *,
    U=16,
    fN=0.4,
    fC=0.9,
    max_steps=100_000,
    seed=1,
    p_count_signal=0.0,
    p_flip_signal=0.0,
):
    d = int(d)
    shots = int(shots)
    U = int(U)
    max_steps = int(max_steps)
    seed = int(seed)

    if shots <= 0:
        raise ValueError(
            "shots must be positive"
        )

    if max_steps <= 0:
        raise ValueError(
            "max_steps must be positive"
        )

    if not 0.0 <= p <= 1.0:
        raise ValueError(
            "p must lie in [0,1]"
        )

    if not 0.0 <= p_count_signal <= 1.0:
        raise ValueError(
            "p_count_signal must lie in [0,1]"
        )

    if not 0.0 <= p_flip_signal <= 1.0:
        raise ValueError(
            "p_flip_signal must lie in [0,1]"
        )

    geometry = _build_geometry(
        d,
        U,
        fN,
        fC,
    )

    p_threshold = (
        _probability_threshold(p)
    )

    count_signal_threshold = (
        _probability_threshold(
            p_count_signal
        )
    )

    flip_signal_threshold = (
        _probability_threshold(
            p_flip_signal
        )
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
            count_signal_threshold,
            flip_signal_threshold,

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
