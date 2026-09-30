"""
Minimal reference implementation of Harrington1D.

The repetition code is represented directly by

    qubits[i] = True / False

with syndrome

    s[i] = qubits[i] XOR qubits[i+1].

The decoder requires d = 3^m.
"""

from dataclasses import dataclass, field


LEFT = 0
RIGHT = 1
CENTER = 2
NONE = -1


# ============================================================
# Repetition code
# ============================================================

def syndrome(qubits):
    d = len(qubits)

    return [
        qubits[i] ^ qubits[(i + 1) % d]
        for i in range(d)
    ]


def has_logical_error(qubits):
    return sum(qubits) > len(qubits) // 2


def apply_corrections(qubits, corrections):
    d = len(qubits)
    new_qubits = qubits.copy()

    for i, direction in enumerate(corrections):
        if direction == LEFT:
            new_qubits[i] ^= True

        elif direction == RIGHT:
            new_qubits[(i + 1) % d] ^= True

    return new_qubits


# ============================================================
# Local Harrington rule
# ============================================================

def harrington_rule(role, left, center, right):
    if role == CENTER or not center:
        return NONE

    if role == LEFT:
        return LEFT if left else RIGHT

    if role == RIGHT:
        return NONE if right else LEFT

    return NONE


# ============================================================
# Hierarchy geometry
# ============================================================

def role_from_coordinate(x):
    """Map colony coordinate 0,1,2 -> LEFT,CENTER,RIGHT."""

    if x == 0:
        return LEFT

    if x == 1:
        return CENTER

    if x == 2:
        return RIGHT

    raise ValueError(
        "coordinate must be 0, 1, or 2"
    )


def hierarchy_depth(d):
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


def level_role(site, level):
    """
    Role of a physical site at hierarchy level `level`.

    level = 0:
        ordinary 3-site colony.

    level >= 1:
        only hierarchy-center sites participate;
        all other sites have role NONE.
    """

    if level == 0:
        return role_from_coordinate(
            site % 3
        )

    scale = 3**level
    offset = (scale - 1) // 2

    if (site - offset) % scale != 0:
        return NONE

    coarse_site = (
        (site - offset)
        // scale
    )

    return role_from_coordinate(
        coarse_site % 3
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
        default_factory=lambda: [
            False,
            False,
        ]
    )

    flip_signal: list = field(
        default_factory=lambda: [
            False,
            False,
        ]
    )

    count: list = field(
        default_factory=lambda: [
            0,
            0,
            0,
        ]
    )

    def reset(self):
        self.count_signal = [
            False,
            False,
        ]

        self.flip_signal = [
            False,
            False,
        ]

        self.count = [
            0,
            0,
            0,
        ]


# ============================================================
# Harrington decoder
# ============================================================

class Harrington1D:

    def __init__(
        self,
        d,
        U,
        fN,
        fC,
    ):
        self.d = d
        self.U = U
        self.fN = fN
        self.fC = fC

        depth = hierarchy_depth(d)

        self.memory = []

        for i in range(d):
            levels = []

            for k in range(1, depth):
                levels.append(
                    Memory(
                        role=level_role(
                            i,
                            k,
                        ),
                        U=U**k,
                        Q=3**k,
                    )
                )

            self.memory.append(
                levels
            )


    def step(self, defects):
        """
        Perform one Harrington1D CA step and return one
        correction direction for every physical cell.
        """

        d = self.d
        n_levels = (
            len(self.memory[0])
            if self.memory
            else 0
        )

        # ----------------------------------------------------
        # 1. Acquire hierarchy signals
        # ----------------------------------------------------

        incoming_count = [
            [
                [False, False]
                for _ in range(n_levels)
            ]
            for _ in range(d)
        ]

        incoming_flip = [
            [
                [False, False]
                for _ in range(n_levels)
            ]
            for _ in range(d)
        ]

        for i in range(d):
            left = (i - 1) % d
            right = (i + 1) % d

            for k in range(n_levels):

                # LEFT-moving signals arrive from the right.
                incoming_count[i][k][LEFT] = (
                    self.memory[right][k]
                    .count_signal[LEFT]
                )

                incoming_flip[i][k][LEFT] = (
                    self.memory[right][k]
                    .flip_signal[LEFT]
                )

                # RIGHT-moving signals arrive from the left.
                incoming_count[i][k][RIGHT] = (
                    self.memory[left][k]
                    .count_signal[RIGHT]
                )

                incoming_flip[i][k][RIGHT] = (
                    self.memory[left][k]
                    .flip_signal[RIGHT]
                )

        # ----------------------------------------------------
        # 2. Update hierarchy memories
        # ----------------------------------------------------

        for i in range(d):
            for k, mem in enumerate(
                self.memory[i]
            ):

                mem.age = (
                    mem.age + 1
                ) % (
                    mem.U + mem.Q
                )

                if mem.role != NONE:

                    # Emit current defect in both directions.
                    mem.count_signal[LEFT] = (
                        defects[i]
                    )

                    mem.count_signal[RIGHT] = (
                        defects[i]
                    )

                    # Accumulate evidence.
                    mem.count[CENTER] += (
                        defects[i]
                    )

                    mem.count[LEFT] += (
                        incoming_count[i][k][RIGHT]
                    )

                    mem.count[RIGHT] += (
                        incoming_count[i][k][LEFT]
                    )

                else:
                    # Non-centers relay incoming signals.
                    mem.count_signal = (
                        incoming_count[i][k].copy()
                    )

                    mem.flip_signal = (
                        incoming_flip[i][k].copy()
                    )

        # ----------------------------------------------------
        # 3. Choose corrections
        # ----------------------------------------------------

        corrections = [
            NONE
            for _ in range(d)
        ]

        for i in range(d):

            # Higher hierarchy levels have priority.
            for k in range(
                n_levels - 1,
                -1,
                -1,
            ):
                mem = self.memory[i][k]

                # Threshold accumulated evidence.
                if (
                    mem.role != NONE
                    and mem.age == mem.U - 1
                ):
                    coarse_left = (
                        mem.count[LEFT]
                        >= self.fN * mem.U
                    )

                    coarse_center = (
                        mem.count[CENTER]
                        >= self.fC * mem.U
                    )

                    coarse_right = (
                        mem.count[RIGHT]
                        >= self.fN * mem.U
                    )

                    direction = (
                        harrington_rule(
                            mem.role,
                            coarse_left,
                            coarse_center,
                            coarse_right,
                        )
                    )

                    if direction != NONE:
                        mem.flip_signal[
                            direction
                        ] = True

                # Release propagated hierarchy correction.
                if (
                    mem.age
                    == mem.U + mem.Q - 1
                ):
                    flip_left = (
                        mem.flip_signal[LEFT]
                    )

                    flip_right = (
                        mem.flip_signal[RIGHT]
                    )

                    mem.reset()

                    if flip_left:
                        corrections[i] = LEFT
                        break

                    if flip_right:
                        corrections[i] = RIGHT
                        break

            # Fallback physical rule.
            if corrections[i] == NONE:
                corrections[i] = (
                    harrington_rule(
                        level_role(i, 0),
                        defects[(i - 1) % d],
                        defects[i],
                        defects[(i + 1) % d],
                    )
                )

        return corrections


# ============================================================
# Complete code-capacity decoder
# ============================================================

def decode_code_capacity(
    qubits,
    U,
    fN,
    fC,
    max_steps,
):
    """
    Decode one fixed physical error configuration.

    Returns:
        final_qubits, resolved, steps
    """

    qubits = qubits.copy()

    decoder = Harrington1D(
        d=len(qubits),
        U=U,
        fN=fN,
        fC=fC,
    )

    for t in range(max_steps):

        defects = syndrome(
            qubits
        )

        if not any(defects):
            return (
                qubits,
                True,
                t,
            )

        corrections = decoder.step(
            defects
        )

        qubits = apply_corrections(
            qubits,
            corrections,
        )

    return (
        qubits,
        False,
        max_steps,
    )
