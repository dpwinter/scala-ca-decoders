"""
Minimal reference implementation of Harrington2D.

Physical X errors live on toric-code edges:

    horizontal[y][x]
    vertical[y][x]

with vertex syndrome

    s[y][x]
      = horizontal[y][x]
      ^ horizontal[y][x+1]
      ^ vertical[y-1][x]
      ^ vertical[y][x].

Harrington2D requires d = 3^m.

The implementation follows the paper dynamics:
    - U = 16
    - Q = 3
    - fC = 0.9
    - fN = 0.4
    - 8 CountSignals per hierarchy level
    - 4 cardinal FlipSignals per hierarchy level
"""

from dataclasses import dataclass, field

import numpy as np


# ============================================================
# Locations
#
#       NW  N  NE
#        W  C   E
#       SW  S  SE
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


CARDINAL = (N, W, E, S)
NEIGHBORS = (N, W, E, S, NW, NE, SW, SE)


DELTAS = {
    N: (-1, 0),
    W: (0, -1),
    E: (0, 1),
    S: (1, 0),
    NW: (-1, -1),
    NE: (-1, 1),
    SW: (1, -1),
    SE: (1, 1),
}


OPPOSITE = {
    N: S,
    W: E,
    E: W,
    S: N,
    NW: SE,
    NE: SW,
    SW: NE,
    SE: NW,
}


# ============================================================
# Toric code
# ============================================================

def zeros(d):
    return [
        [False for _ in range(d)]
        for _ in range(d)
    ]


def syndrome(horizontal, vertical):
    d = len(horizontal)

    defects = zeros(d)

    for y in range(d):
        ym = (y - 1) % d

        for x in range(d):
            xp = (x + 1) % d

            defects[y][x] = (
                horizontal[y][x]
                ^ horizontal[y][xp]
                ^ vertical[ym][x]
                ^ vertical[y][x]
            )

    return defects


def has_logical_error(horizontal, vertical):
    """
    Exact logical check matching toric_code_z_logicals().
    """

    d = len(horizontal)

    logical_h = False
    logical_v = False

    for y in range(d):
        logical_h ^= horizontal[y][0]

    for x in range(d):
        logical_v ^= vertical[0][x]

    return logical_h or logical_v


def apply_corrections(
    horizontal,
    vertical,
    corrections,
):
    """
    Apply all cell correction requests modulo 2.

    N -> vertical[y-1][x]
    W -> horizontal[y][x]
    E -> horizontal[y][x+1]
    S -> vertical[y][x]
    """

    d = len(horizontal)

    h = [
        row.copy()
        for row in horizontal
    ]

    v = [
        row.copy()
        for row in vertical
    ]

    for y in range(d):
        ym = (y - 1) % d

        for x in range(d):
            xp = (x + 1) % d

            direction = corrections[y][x]

            if direction == N:
                v[ym][x] ^= True

            elif direction == W:
                h[y][x] ^= True

            elif direction == E:
                h[y][xp] ^= True

            elif direction == S:
                v[y][x] ^= True

    return h, v


# ============================================================
# Harrington local rule
#
# Each address has an ordered priority list.
# The first matching neighboring defect determines the action.
# ============================================================

RULES = {
    NW: (
        (N, NONE),
        (W, W),
        (E, E),
        (S, S),
        (NW, W),
        (NE, NONE),
        (SW, W),
        (None, E),
    ),

    N: (
        (N, NONE),
        (W, NONE),
        (E, NONE),
        (S, S),
        (NW, NONE),
        (NE, NONE),
        (None, S),
    ),

    NE: (
        (N, NONE),
        (E, NONE),
        (W, W),
        (S, S),
        (NW, NONE),
        (NE, NONE),
        (SE, NONE),
        (None, W),
    ),

    W: (
        (W, W),
        (N, NONE),
        (S, NONE),
        (E, E),
        (NW, W),
        (SW, W),
        (None, E),
    ),

    E: (
        (E, NONE),
        (N, NONE),
        (S, NONE),
        (W, W),
        (NE, NONE),
        (SE, NONE),
        (None, W),
    ),

    SW: (
        (W, W),
        (S, S),
        (N, N),
        (E, E),
        (NW, W),
        (SW, S),
        (SE, S),
        (None, E),
    ),

    S: (
        (S, S),
        (W, NONE),
        (E, NONE),
        (N, N),
        (SE, S),
        (SW, S),
        (None, N),
    ),

    SE: (
        (E, NONE),
        (S, S),
        (W, W),
        (N, N),
        (NE, NONE),
        (SE, NONE),
        (SW, S),
        (None, W),
    ),
}


def harrington_rule(role, defects):
    """
    defects is indexed by N,W,E,S,NW,NE,SW,SE,C.
    """

    if (
        role == C
        or role == NONE
        or not defects[C]
    ):
        return NONE

    for neighbor, direction in RULES[role]:

        if neighbor is None:
            return direction

        if defects[neighbor]:
            return direction

    return NONE


# ============================================================
# Hierarchy geometry
# ============================================================

def hierarchy_depth(d):
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


def role_from_coordinates(row, col):
    grid = (
        (NW, N, NE),
        (W, C, E),
        (SW, S, SE),
    )

    return grid[row][col]


def level_role(row, col, level):
    """
    level = 0:
        physical 3x3 colony.

    level >= 1:
        only hierarchy representatives have a role;
        all other cells return NONE.
    """

    if level == 0:
        return role_from_coordinates(
            row % 3,
            col % 3,
        )

    scale = 3**level
    offset = (scale - 1) // 2

    dr = row - offset
    dc = col - offset

    if (
        dr % scale != 0
        or dc % scale != 0
    ):
        return NONE

    coarse_row = dr // scale
    coarse_col = dc // scale

    return role_from_coordinates(
        coarse_row % 3,
        coarse_col % 3,
    )


# ============================================================
# One hierarchy level
# ============================================================

@dataclass
class Memory:
    role: int
    U: int
    Q: int

    age: int = 0

    count_signal: list = field(
        default_factory=lambda: [False] * 8
    )

    flip_signal: list = field(
        default_factory=lambda: [False] * 4
    )

    count: list = field(
        default_factory=lambda: [0] * 9
    )

    def reset(self):
        self.count_signal = [False] * 8
        self.flip_signal = [False] * 4
        self.count = [0] * 9


# ============================================================
# Harrington2D decoder
# ============================================================

class Harrington2D:

    def __init__(
        self,
        d,
        U=16,
        fN=0.4,
        fC=0.9,
        seed=12345,
    ):
        self.d = d
        self.U = U
        self.fN = fN
        self.fC = fC

        self.rng = np.random.default_rng(
            seed
        )

        depth = hierarchy_depth(d)

        self.memory = []

        for y in range(d):
            row = []

            for x in range(d):
                levels = []

                for k in range(1, depth):
                    levels.append(
                        Memory(
                            role=level_role(
                                y,
                                x,
                                k,
                            ),
                            U=U**k,
                            Q=3**k,
                        )
                    )

                row.append(levels)

            self.memory.append(row)


    def step(
        self,
        defects,
        p_signal=0.0,
    ):
        """
        Perform one synchronous Harrington2D CA step.

        Returns a d x d array of correction directions.
        """

        d = self.d

        n_levels = (
            len(self.memory[0][0])
            if self.memory
            else 0
        )

        # ----------------------------------------------------
        # 1. Acquire incoming signals synchronously
        # ----------------------------------------------------

        incoming_count = [
            [
                [
                    [False] * 8
                    for _ in range(n_levels)
                ]
                for _ in range(d)
            ]
            for _ in range(d)
        ]

        incoming_flip = [
            [
                [
                    [False] * 4
                    for _ in range(n_levels)
                ]
                for _ in range(d)
            ]
            for _ in range(d)
        ]

        for y in range(d):
            for x in range(d):

                for k in range(n_levels):

                    for direction in NEIGHBORS:
                        dy, dx = DELTAS[
                            OPPOSITE[direction]
                        ]

                        sy = (y + dy) % d
                        sx = (x + dx) % d

                        incoming_count[y][x][k][
                            direction
                        ] = (
                            self.memory[sy][sx][k]
                            .count_signal[direction]
                        )

                    for direction in CARDINAL:
                        dy, dx = DELTAS[
                            OPPOSITE[direction]
                        ]

                        sy = (y + dy) % d
                        sx = (x + dx) % d

                        incoming_flip[y][x][k][
                            direction
                        ] = (
                            self.memory[sy][sx][k]
                            .flip_signal[direction]
                        )

        # ----------------------------------------------------
        # 2. Signal noise + memory update
        # ----------------------------------------------------

        for y in range(d):
            for x in range(d):

                for k, mem in enumerate(
                    self.memory[y][x]
                ):
                    inc_count = (
                        incoming_count[y][x][k]
                    )

                    # Signal noise acts only on incoming
                    # CountSignals, exactly as in the C++.
                    if p_signal > 0.0:
                        for direction in NEIGHBORS:
                            if (
                                self.rng.random()
                                < p_signal
                            ):
                                inc_count[
                                    direction
                                ] ^= True

                    mem.age = (
                        mem.age + 1
                    ) % (
                        mem.U
                        + mem.Q
                        + 1
                    )

                    if mem.role != NONE:

                        # Representatives broadcast their
                        # physical defect in all directions.
                        for direction in NEIGHBORS:
                            mem.count_signal[
                                direction
                            ] = defects[y][x]

                        mem.count[C] += defects[y][x]

                        # Incoming direction d contributes to
                        # the count associated with the
                        # opposite neighbor.
                        for direction in NEIGHBORS:
                            mem.count[
                                direction
                            ] += inc_count[
                                OPPOSITE[direction]
                            ]

                    else:
                        # Non-representatives relay signals.
                        mem.count_signal = (
                            inc_count.copy()
                        )

                        mem.flip_signal = (
                            incoming_flip[y][x][k]
                            .copy()
                        )

        # ----------------------------------------------------
        # 3. Determine corrections
        # ----------------------------------------------------

        corrections = [
            [NONE for _ in range(d)]
            for _ in range(d)
        ]

        for y in range(d):
            for x in range(d):

                direction = NONE

                # Higher hierarchy levels take priority.
                for k in range(
                    n_levels - 1,
                    -1,
                    -1,
                ):
                    mem = self.memory[y][x][k]

                    # Threshold accumulated counts.
                    if (
                        mem.role != NONE
                        and mem.age == mem.U
                    ):
                        coarse = [
                            False for _ in range(9)
                        ]

                        for loc in range(9):
                            threshold = (
                                self.fC
                                if loc == C
                                else self.fN
                            )

                            coarse[loc] = (
                                mem.count[loc]
                                >= threshold * mem.U
                            )

                            # The C++ clears counters here.
                            mem.count[loc] = 0

                        move = harrington_rule(
                            mem.role,
                            coarse,
                        )

                        if move != NONE:
                            mem.flip_signal[
                                move
                            ] = True

                    # Release hierarchical correction.
                    if (
                        mem.age
                        == mem.U + mem.Q
                    ):
                        flip = (
                            mem.flip_signal.copy()
                        )

                        mem.reset()

                        for move in CARDINAL:
                            if flip[move]:
                                direction = move
                                break

                        if direction != NONE:
                            break

                # Physical fallback rule.
                if direction == NONE:

                    local = [
                        False for _ in range(9)
                    ]

                    local[C] = defects[y][x]

                    for loc in NEIGHBORS:
                        dy, dx = DELTAS[loc]

                        local[loc] = defects[
                            (y + dy) % d
                        ][
                            (x + dx) % d
                        ]

                    direction = harrington_rule(
                        level_role(y, x, 0),
                        local,
                    )

                corrections[y][x] = direction

        return corrections


# ============================================================
# Complete code-capacity decoder
# ============================================================

def decode_code_capacity(
    horizontal,
    vertical,
    *,
    U=16,
    fN=0.4,
    fC=0.9,
    seed=12345,
    max_steps=None,
):
    """
    Decode one fixed toric-code X-error configuration.

    The paper prescription is used:
        run until the syndrome vanishes.

    max_steps is optional and intended only as a safety guard
    during testing.

    Returns:
        final_horizontal,
        final_vertical,
        resolved,
        steps
    """

    h = [
        row.copy()
        for row in horizontal
    ]

    v = [
        row.copy()
        for row in vertical
    ]

    decoder = Harrington2D(
        d=len(h),
        U=U,
        fN=fN,
        fC=fC,
        seed=seed,
    )

    t = 0

    while True:

        defects = syndrome(
            h,
            v,
        )

        if not any(
            any(row)
            for row in defects
        ):
            return (
                h,
                v,
                True,
                t,
            )

        if (
            max_steps is not None
            and t >= max_steps
        ):
            return (
                h,
                v,
                False,
                t,
            )

        corrections = decoder.step(
            defects,
            p_signal=0.0,
        )

        h, v = apply_corrections(
            h,
            v,
            corrections,
        )

        t += 1
