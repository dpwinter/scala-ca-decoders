"""
Tests for the minimal Harrington2D reference implementation.

Run from the project root:

    python -m scripts.test_harrington2d
"""

import itertools

import numpy as np
from scipy.sparse import block_diag, csc_matrix, eye, hstack, kron

from src.harrington2d import (
    N,
    W,
    E,
    S,
    NW,
    NE,
    SW,
    SE,
    C,
    NONE,
    CARDINAL,
    NEIGHBORS,
    Harrington2D,
    syndrome,
    has_logical_error,
    apply_corrections,
    harrington_rule,
    hierarchy_depth,
    level_role,
    role_from_coordinates,
    zeros,
    decode_code_capacity,
)


U = 16
FN = 0.4
FC = 0.9


# ============================================================
# Legacy toric-code construction
# ============================================================

def repetition_code(n):
    rows = []
    cols = []

    for i in range(n):
        rows.extend([i, i])
        cols.extend([i, (i + 1) % n])

    data = np.ones(
        2 * n,
        dtype=np.uint8,
    )

    return csc_matrix(
        (data, (rows, cols)),
        shape=(n, n),
    )


def legacy_stabilisers(d):
    Hr = repetition_code(d)

    H = hstack(
        [
            kron(eye(d), Hr),
            kron(Hr.T, eye(d)),
        ],
        dtype=np.uint8,
    )

    H.data %= 2
    H.eliminate_zeros()

    return csc_matrix(H)


def legacy_logicals(d):
    H1 = csc_matrix(
        ([1], ([0], [0])),
        shape=(1, d),
        dtype=np.uint8,
    )

    H0 = csc_matrix(
        np.ones(
            (1, d),
            dtype=np.uint8,
        )
    )

    logicals = block_diag(
        [
            kron(H0, H1),
            kron(H1, H0),
        ]
    )

    return csc_matrix(logicals)


def flatten_frame(h, v):
    return np.concatenate(
        [
            np.asarray(h, dtype=np.uint8).ravel(),
            np.asarray(v, dtype=np.uint8).ravel(),
        ]
    )


# ============================================================
# Independent C++ local rule
# ============================================================

def legacy_rule(addr, d):
    """
    Literal transcription of Harrington2DCell::harringtonRule().

    This is intentionally independent of RULES in src/harrington2d.py.
    """

    if addr == C or not d[C]:
        return NONE

    if addr == NW:
        if d[N]:
            return NONE
        elif d[W]:
            return W
        elif d[E]:
            return E
        elif d[S]:
            return S
        elif d[NW]:
            return W
        elif d[NE]:
            return NONE
        elif d[SW]:
            return W
        else:
            return E

    if addr == N:
        if d[N]:
            return NONE
        elif d[W]:
            return NONE
        elif d[E]:
            return NONE
        elif d[S]:
            return S
        elif d[NW]:
            return NONE
        elif d[NE]:
            return NONE
        else:
            return S

    if addr == NE:
        if d[N]:
            return NONE
        elif d[E]:
            return NONE
        elif d[W]:
            return W
        elif d[S]:
            return S
        elif d[NW]:
            return NONE
        elif d[NE]:
            return NONE
        elif d[SE]:
            return NONE
        else:
            return W

    if addr == W:
        if d[W]:
            return W
        elif d[N]:
            return NONE
        elif d[S]:
            return NONE
        elif d[E]:
            return E
        elif d[NW]:
            return W
        elif d[SW]:
            return W
        else:
            return E

    if addr == E:
        if d[E]:
            return NONE
        elif d[N]:
            return NONE
        elif d[S]:
            return NONE
        elif d[W]:
            return W
        elif d[NE]:
            return NONE
        elif d[SE]:
            return NONE
        else:
            return W

    if addr == SW:
        if d[W]:
            return W
        elif d[S]:
            return S
        elif d[N]:
            return N
        elif d[E]:
            return E
        elif d[NW]:
            return W
        elif d[SW]:
            return S
        elif d[SE]:
            return S
        else:
            return E

    if addr == S:
        if d[S]:
            return S
        elif d[W]:
            return NONE
        elif d[E]:
            return NONE
        elif d[N]:
            return N
        elif d[SE]:
            return S
        elif d[SW]:
            return S
        else:
            return N

    if addr == SE:
        if d[E]:
            return NONE
        elif d[S]:
            return S
        elif d[W]:
            return W
        elif d[N]:
            return N
        elif d[NE]:
            return NONE
        elif d[SE]:
            return NONE
        elif d[SW]:
            return S
        else:
            return W

    return NONE


# ============================================================
# Toric-code convention
# ============================================================

def test_syndrome_matches_legacy():
    rng = np.random.default_rng(12345)

    for d in (3, 5, 7):
        H = legacy_stabilisers(d)

        for _ in range(100):
            h = (
                rng.random((d, d)) < 0.3
            ).tolist()

            v = (
                rng.random((d, d)) < 0.3
            ).tolist()

            expected = np.asarray(
                (H @ flatten_frame(h, v)) % 2
            ).ravel().astype(bool)

            actual = np.asarray(
                syndrome(h, v),
                dtype=bool,
            ).ravel()

            assert np.array_equal(
                actual,
                expected,
            ), f"syndrome mismatch for d={d}"


def test_logicals_match_legacy():
    rng = np.random.default_rng(23456)

    for d in (3, 5, 7):
        logicals = legacy_logicals(d)

        for _ in range(100):
            h = (
                rng.random((d, d)) < 0.3
            ).tolist()

            v = (
                rng.random((d, d)) < 0.3
            ).tolist()

            expected = bool(
                np.any(
                    (
                        logicals
                        @ flatten_frame(h, v)
                    ) % 2
                )
            )

            actual = has_logical_error(
                h,
                v,
            )

            assert actual == expected


# ============================================================
# Physical correction convention
# ============================================================

def one_correction(
    d,
    y,
    x,
    direction,
):
    h = zeros(d)
    v = zeros(d)

    corrections = [
        [NONE for _ in range(d)]
        for _ in range(d)
    ]

    corrections[y][x] = direction

    return apply_corrections(
        h,
        v,
        corrections,
    )


def test_correction_north():
    d = 5

    h, v = one_correction(
        d,
        2,
        3,
        N,
    )

    assert not any(
        any(row)
        for row in h
    )

    assert v[1][3]

    assert sum(
        sum(row)
        for row in v
    ) == 1


def test_correction_west():
    d = 5

    h, v = one_correction(
        d,
        2,
        3,
        W,
    )

    assert h[2][3]

    assert sum(
        sum(row)
        for row in h
    ) == 1

    assert not any(
        any(row)
        for row in v
    )


def test_correction_east():
    d = 5

    h, v = one_correction(
        d,
        2,
        3,
        E,
    )

    assert h[2][4]

    assert sum(
        sum(row)
        for row in h
    ) == 1


def test_correction_south():
    d = 5

    h, v = one_correction(
        d,
        2,
        3,
        S,
    )

    assert v[2][3]

    assert sum(
        sum(row)
        for row in v
    ) == 1


def test_periodic_corrections():
    d = 5

    h, v = one_correction(
        d,
        0,
        4,
        N,
    )

    assert v[4][4]

    h, v = one_correction(
        d,
        2,
        4,
        E,
    )

    assert h[2][0]


def test_correction_cancellation():
    """
    Two cells correcting the same edge cancel modulo 2.
    """

    d = 5

    h = zeros(d)
    v = zeros(d)

    corrections = [
        [NONE for _ in range(d)]
        for _ in range(d)
    ]

    corrections[2][2] = E
    corrections[2][3] = W

    h2, v2 = apply_corrections(
        h,
        v,
        corrections,
    )

    assert not any(
        any(row)
        for row in h2
    )

    assert not any(
        any(row)
        for row in v2
    )


# ============================================================
# Full local rule
# ============================================================

def test_local_rule_exhaustive():
    """
    Compare the compact table-driven Python rule against
    a literal transcription of the C++ rule.

    9 roles x 2^9 local defect patterns.
    """

    roles = [
        N,
        W,
        E,
        S,
        NW,
        NE,
        SW,
        SE,
        C,
        NONE,
    ]

    for role in roles:

        for bits in itertools.product(
            (False, True),
            repeat=9,
        ):
            defects = list(bits)

            expected = legacy_rule(
                role,
                defects,
            )

            actual = harrington_rule(
                role,
                defects,
            )

            assert actual == expected, (
                f"role={role}, "
                f"defects={defects}: "
                f"{actual} != {expected}"
            )


# ============================================================
# Hierarchy geometry
# ============================================================

def test_hierarchy_depth():
    assert hierarchy_depth(3) == 1
    assert hierarchy_depth(9) == 2
    assert hierarchy_depth(27) == 3
    assert hierarchy_depth(81) == 4


def test_non_power_of_three_rejected():
    try:
        hierarchy_depth(12)

    except ValueError:
        return

    raise AssertionError(
        "non-power-of-three distance accepted"
    )


def test_level_zero_roles():
    expected = [
        [NW, N, NE],
        [W, C, E],
        [SW, S, SE],
    ]

    actual = [
        [
            level_role(y, x, 0)
            for x in range(3)
        ]
        for y in range(3)
    ]

    assert actual == expected


def test_level_one_geometry_d9():
    expected_positions = {
        (1, 1): NW,
        (1, 4): N,
        (1, 7): NE,

        (4, 1): W,
        (4, 4): C,
        (4, 7): E,

        (7, 1): SW,
        (7, 4): S,
        (7, 7): SE,
    }

    for y in range(9):
        for x in range(9):

            expected = expected_positions.get(
                (y, x),
                NONE,
            )

            actual = level_role(
                y,
                x,
                1,
            )

            assert actual == expected, (
                f"({y},{x}): "
                f"{actual} != {expected}"
            )


# ============================================================
# Memory construction / timing
# ============================================================

def test_memory_parameters_d9():
    decoder = Harrington2D(
        9,
        U=U,
        fN=FN,
        fC=FC,
    )

    for y in range(9):
        for x in range(9):

            assert (
                len(decoder.memory[y][x])
                == 1
            )

            mem = decoder.memory[y][x][0]

            assert mem.U == 16
            assert mem.Q == 3
            assert mem.age == 0


def test_memory_parameters_d27():
    decoder = Harrington2D(
        27,
        U=U,
        fN=FN,
        fC=FC,
    )

    mem = decoder.memory[0][0]

    assert len(mem) == 2

    assert mem[0].U == 16
    assert mem[0].Q == 3

    assert mem[1].U == 256
    assert mem[1].Q == 9


# ============================================================
# Signal propagation
# ============================================================

def empty_defects(d):
    return zeros(d)


def test_count_signal_propagation():
    """
    Check all eight directions.

    A signal labelled direction D moves one physical site
    in direction D.
    """

    d = 9
    source = (3, 3)

    for direction in NEIGHBORS:

        decoder = Harrington2D(
            d,
            seed=1,
        )

        sy, sx = source

        decoder.memory[
            sy
        ][
            sx
        ][0].count_signal[
            direction
        ] = True

        defects = empty_defects(d)

        decoder.step(
            defects,
            p_signal=0.0,
        )

        if direction == N:
            target = (2, 3)
        elif direction == W:
            target = (3, 2)
        elif direction == E:
            target = (3, 4)
        elif direction == S:
            target = (4, 3)
        elif direction == NW:
            target = (2, 2)
        elif direction == NE:
            target = (2, 4)
        elif direction == SW:
            target = (4, 2)
        elif direction == SE:
            target = (4, 4)

        ty, tx = target

        # Avoid representatives because they broadcast instead
        # of relaying CountSignals.
        if (
            decoder.memory[ty][tx][0].role
            == NONE
        ):
            assert decoder.memory[
                ty
            ][
                tx
            ][0].count_signal[
                direction
            ]


def test_flip_signal_propagation():
    """
    FlipSignals propagate only in cardinal directions.
    """

    d = 9
    source = (3, 3)

    targets = {
        N: (2, 3),
        W: (3, 2),
        E: (3, 4),
        S: (4, 3),
    }

    for direction in CARDINAL:

        decoder = Harrington2D(
            d,
            seed=1,
        )

        sy, sx = source

        decoder.memory[
            sy
        ][
            sx
        ][0].flip_signal[
            direction
        ] = True

        decoder.step(
            empty_defects(d),
            p_signal=0.0,
        )

        ty, tx = targets[
            direction
        ]

        assert decoder.memory[
            ty
        ][
            tx
        ][0].flip_signal[
            direction
        ]


# ============================================================
# Representative broadcasting / counters
# ============================================================

def test_representative_broadcast():
    d = 9

    decoder = Harrington2D(
        d,
        seed=1,
    )

    defects = empty_defects(d)

    # Level-1 NW representative.
    defects[1][1] = True

    decoder.step(
        defects,
        p_signal=0.0,
    )

    mem = decoder.memory[1][1][0]

    assert mem.role == NW

    assert all(
        mem.count_signal
    )

    assert mem.count[C] == 1


# ============================================================
# Signal noise
# ============================================================

def test_signal_noise_probability_one():
    """
    p_signal=1 flips every incoming CountSignal.

    At a non-representative site with initially empty signals,
    all eight propagated CountSignals therefore become one.
    """

    d = 9

    decoder = Harrington2D(
        d,
        seed=1,
    )

    decoder.step(
        empty_defects(d),
        p_signal=1.0,
    )

    # (3,3) is not a level-1 representative.
    mem = decoder.memory[3][3][0]

    assert mem.role == NONE
    assert all(mem.count_signal)

    # FlipSignals are not affected by signal noise.
    assert not any(
        mem.flip_signal
    )


# ============================================================
# Hierarchy threshold / release timing
# ============================================================

def test_hierarchy_threshold():
    """
    Force a level-1 NW representative to threshold to the
    default E correction.
    """

    d = 9

    decoder = Harrington2D(
        d,
        seed=1,
    )

    mem = decoder.memory[1][1][0]

    assert mem.role == NW

    # step() increments age first.
    mem.age = mem.U - 1

    # Coarse central defect only:
    # NW rule defaults to E.
    mem.count[C] = int(
        np.ceil(FC * mem.U)
    )

    decoder.step(
        empty_defects(d),
        p_signal=0.0,
    )

    assert mem.age == mem.U
    assert mem.flip_signal[E]


def test_hierarchy_release():
    """
    A stored FlipSignal is released at age U+Q.
    """

    d = 9

    decoder = Harrington2D(
        d,
        seed=1,
    )

    mem = decoder.memory[1][1][0]

    mem.age = (
        mem.U
        + mem.Q
        - 1
    )

    mem.flip_signal[E] = True

    corrections = decoder.step(
        empty_defects(d),
        p_signal=0.0,
    )

    assert corrections[1][1] == E

    assert mem.age == (
        mem.U + mem.Q
    )

    assert not any(
        mem.flip_signal
    )

    assert not any(
        mem.count_signal
    )

    assert not any(
        mem.count
    )


# ============================================================
# Small code-capacity regression
# ============================================================

def test_zero_error_code_capacity():
    for d in (3, 9):

        h = zeros(d)
        v = zeros(d)

        h2, v2, resolved, steps = (
            decode_code_capacity(
                h,
                v,
                max_steps=1000,
            )
        )

        assert resolved
        assert steps == 0

        assert h2 == h
        assert v2 == v

        assert not has_logical_error(
            h2,
            v2,
        )


def test_all_single_errors_d3():
    """
    Every one of the 2*d^2 = 18 single-edge errors should
    be corrected without a logical error.
    """

    d = 3

    # Horizontal edges
    for y in range(d):
        for x in range(d):

            h = zeros(d)
            v = zeros(d)

            h[y][x] = True

            h2, v2, resolved, _ = (
                decode_code_capacity(
                    h,
                    v,
                    max_steps=1000,
                )
            )

            assert resolved, (
                f"horizontal error "
                f"({y},{x}) did not resolve"
            )

            assert not any(
                any(row)
                for row in syndrome(
                    h2,
                    v2,
                )
            )

            assert not has_logical_error(
                h2,
                v2,
            )

    # Vertical edges
    for y in range(d):
        for x in range(d):

            h = zeros(d)
            v = zeros(d)

            v[y][x] = True

            h2, v2, resolved, _ = (
                decode_code_capacity(
                    h,
                    v,
                    max_steps=1000,
                )
            )

            assert resolved, (
                f"vertical error "
                f"({y},{x}) did not resolve"
            )

            assert not any(
                any(row)
                for row in syndrome(
                    h2,
                    v2,
                )
            )

            assert not has_logical_error(
                h2,
                v2,
            )


# ============================================================
# Run
# ============================================================

def run_tests():
    tests = [
        test_syndrome_matches_legacy,
        test_logicals_match_legacy,

        test_correction_north,
        test_correction_west,
        test_correction_east,
        test_correction_south,
        test_periodic_corrections,
        test_correction_cancellation,

        test_local_rule_exhaustive,

        test_hierarchy_depth,
        test_non_power_of_three_rejected,
        test_level_zero_roles,
        test_level_one_geometry_d9,

        test_memory_parameters_d9,
        test_memory_parameters_d27,

        test_count_signal_propagation,
        test_flip_signal_propagation,
        test_representative_broadcast,

        test_signal_noise_probability_one,

        test_hierarchy_threshold,
        test_hierarchy_release,

        test_zero_error_code_capacity,
        test_all_single_errors_d3,
    ]

    print(
        "Harrington2D reference tests"
    )

    print("=" * 52)

    for test in tests:
        test()

        print(
            f"PASS  {test.__name__}"
        )

    print("=" * 52)

    print(
        f"All {len(tests)} tests passed."
    )


if __name__ == "__main__":
    run_tests()
