"""
Tests for the SCALA2D paper reference implementation.

Run from the project root with:

    python -m scripts.test_scala2d
"""

from src.scala2d import (
    zeros,
    syndrome,
    reset_signals,
    correction,
    step,
    reset_schedule,
    decode_code_capacity,
)


def count(a):
    return sum(sum(row) for row in a)


def empty_state(d):
    h = zeros(d)
    v = zeros(d)
    n, w, e, s = reset_signals(d)

    return h, v, n, w, e, s


# ============================================================
# Syndrome geometry
# ============================================================

def test_horizontal_error_syndrome():
    d = 5

    h = zeros(d)
    v = zeros(d)

    h[2][3] = True

    defect = syndrome(h, v)

    assert count(defect) == 2
    assert defect[2][3]
    assert defect[2][2]


def test_vertical_error_syndrome():
    d = 5

    h = zeros(d)
    v = zeros(d)

    v[2][3] = True

    defect = syndrome(h, v)

    assert count(defect) == 2
    assert defect[2][3]
    assert defect[1][3]


def test_periodic_syndrome():
    d = 5

    h = zeros(d)
    v = zeros(d)

    h[2][0] = True

    defect = syndrome(h, v)

    assert count(defect) == 2
    assert defect[2][0]
    assert defect[2][d - 1]


# ============================================================
# Broadcast and propagation
# ============================================================

def test_broadcast():
    d = 5

    h, v, n, w, e, s = empty_state(d)

    h[2][2] = True

    _, _, n2, w2, e2, s2 = step(
        h, v, n, w, e, s
    )

    assert n2[1][2]
    assert s2[3][2]
    assert w2[2][1]
    assert e2[2][3]


def test_propagation():
    d = 5

    h, v, n, w, e, s = empty_state(d)

    e[2][2] = True

    _, _, n2, w2, e2, s2 = step(
        h, v, n, w, e, s
    )

    assert e2[2][3]

    assert count(e2) == 1
    assert count(n2) == 0
    assert count(w2) == 0
    assert count(s2) == 0


def test_periodic_propagation():
    d = 5

    h, v, n, w, e, s = empty_state(d)

    e[2][d - 1] = True

    _, _, _, _, e2, _ = step(
        h, v, n, w, e, s
    )

    assert e2[2][0]
    assert count(e2) == 1


# ============================================================
# Reflection
# ============================================================

def test_reflection():
    d = 5

    h, v, n, w, e, s = empty_state(d)

    e[2][1] = True
    s[1][2] = True

    _, _, n2, w2, e2, s2 = step(
        h, v, n, w, e, s
    )

    assert n2[2][2]
    assert w2[2][2]

    assert not e2[2][2]
    assert not s2[2][2]


def test_no_reflection_for_one_signal():
    d = 5

    h, v, n, w, e, s = empty_state(d)

    e[2][1] = True

    _, _, n2, w2, e2, s2 = step(
        h, v, n, w, e, s
    )

    assert e2[2][2]

    assert not n2[2][2]
    assert not w2[2][2]
    assert not s2[2][2]


# ============================================================
# Local correction rule
# ============================================================

def test_correction_requires_defect():
    result = correction(
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    )

    assert result is None


def test_nearest_neighbor_west():
    result = correction(
        True,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    )

    assert result == "W"


def test_nearest_neighbor_north():
    result = correction(
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    )

    assert result == "N"


def test_nearest_neighbor_priority():
    result = correction(
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    )

    assert result == "W"


def test_signal_follow_requires_isolated_defect():
    # Isolated defect with a north signal.
    result = correction(
        True,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    )

    assert result == "S"

    # East neighbor destroys isolation.
    result = correction(
        True,
        False,
        False,
        True,
        False,
        True,
        False,
        False,
        False,
    )

    assert result is None

    # South neighbor destroys isolation.
    result = correction(
        True,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
    )

    assert result is None


# ============================================================
# Exhaustive Signal-Follow truth table
#
# Tuple order:
#
#     (N, W, E, S)
#
# Result is the direction of the corrected qubit.
# ============================================================

def test_all_16_signal_patterns():
    expected = {
        # 0 signals
        (0, 0, 0, 0): None,

        # 1 signal
        (1, 0, 0, 0): "S",
        (0, 1, 0, 0): "E",
        (0, 0, 1, 0): "W",
        (0, 0, 0, 1): "N",

        # 2 signals
        (1, 1, 0, 0): "S",   # NW
        (1, 0, 1, 0): "W",   # NE
        (1, 0, 0, 1): None,  # NS
        (0, 1, 1, 0): None,  # WE
        (0, 1, 0, 1): None,  # WS
        (0, 0, 1, 1): None,  # ES

        # 3 signals
        (1, 1, 1, 0): "S",   # NWE
        (1, 1, 0, 1): "E",   # NWS
        (1, 0, 1, 1): "W",   # NES
        (0, 1, 1, 1): "N",   # WES

        # 4 signals
        (1, 1, 1, 1): None,
    }

    assert len(expected) == 16

    for signals, expected_direction in expected.items():
        n, w, e, s = signals

        result = correction(
            True,
            False,
            False,
            False,
            False,
            bool(n),
            bool(w),
            bool(e),
            bool(s),
        )

        assert result == expected_direction, (
            f"signals N,W,E,S={signals}: "
            f"expected {expected_direction}, "
            f"got {result}"
        )


# ============================================================
# Physical correction geometry
# ============================================================

def test_west_correction_edge():
    d = 5

    h, v, n, w, e, s = empty_state(d)

    h[2][2] = True

    h2, v2, *_ = step(
        h, v, n, w, e, s
    )

    assert count(syndrome(h2, v2)) == 0


def test_north_correction_edge():
    d = 5

    h, v, n, w, e, s = empty_state(d)

    v[2][2] = True

    h2, v2, *_ = step(
        h, v, n, w, e, s
    )

    assert count(syndrome(h2, v2)) == 0


# ============================================================
# Reset schedule
# ============================================================

def test_reset():
    d = 5

    n = [[True] * d for _ in range(d)]
    w = [[True] * d for _ in range(d)]
    e = [[True] * d for _ in range(d)]
    s = [[True] * d for _ in range(d)]

    n, w, e, s = reset_signals(d)

    assert count(n) == 0
    assert count(w) == 0
    assert count(e) == 0
    assert count(s) == 0


def test_reset_schedule():
    assert reset_schedule(3) == [
        1,
        2,
        3,
        2,
        1,
    ]

    assert reset_schedule(5) == [
        1,
        2,
        3,
        4,
        5,
        4,
        3,
        2,
        1,
    ]


def test_total_runtime():
    for d in range(3, 20, 2):
        schedule = reset_schedule(d)

        assert schedule[0] == 1
        assert schedule[-1] == 1
        assert max(schedule) == d
        assert sum(schedule) == d * d


# ============================================================
# Logical sector
# ============================================================

def logical_error(horizontal, vertical):
    d = len(horizontal)

    logical_x = False
    logical_y = False

    for x in range(d):
        logical_x ^= vertical[0][x]

    for y in range(d):
        logical_y ^= horizontal[y][0]

    return logical_x or logical_y


# ============================================================
# End-to-end code-capacity test
# ============================================================

def test_small_code_capacity():
    """
    Run the complete d^2-step decoder on every weight-0 and
    weight-1 physical error for d=3 and d=5.
    """

    for d in (3, 5):
        # Weight 0
        h = zeros(d)
        v = zeros(d)

        h2, v2 = decode_code_capacity(h, v)

        assert count(syndrome(h2, v2)) == 0
        assert not logical_error(h2, v2)

        # Every horizontal weight-1 error
        for y in range(d):
            for x in range(d):
                h = zeros(d)
                v = zeros(d)

                h[y][x] = True

                h2, v2 = decode_code_capacity(h, v)

                assert count(syndrome(h2, v2)) == 0, (
                    f"d={d}, horizontal error "
                    f"({y},{x}) left residual syndrome"
                )

                assert not logical_error(h2, v2), (
                    f"d={d}, horizontal error "
                    f"({y},{x}) caused logical failure"
                )

        # Every vertical weight-1 error
        for y in range(d):
            for x in range(d):
                h = zeros(d)
                v = zeros(d)

                v[y][x] = True

                h2, v2 = decode_code_capacity(h, v)

                assert count(syndrome(h2, v2)) == 0, (
                    f"d={d}, vertical error "
                    f"({y},{x}) left residual syndrome"
                )

                assert not logical_error(h2, v2), (
                    f"d={d}, vertical error "
                    f"({y},{x}) caused logical failure"
                )


# ============================================================
# General toric-code consistency
# ============================================================

def test_even_syndrome_parity():
    for d in (3, 5, 7):
        h = zeros(d)
        v = zeros(d)

        for y in range(d):
            for x in range(d):
                h[y][x] = (
                    (3 * y + 5 * x) % 7
                ) < 3

                v[y][x] = (
                    (5 * y + 2 * x) % 11
                ) < 4

        assert count(syndrome(h, v)) % 2 == 0


# ============================================================
# Run
# ============================================================

def run_tests():
    tests = [
        test_horizontal_error_syndrome,
        test_vertical_error_syndrome,
        test_periodic_syndrome,

        test_broadcast,
        test_propagation,
        test_periodic_propagation,

        test_reflection,
        test_no_reflection_for_one_signal,

        test_correction_requires_defect,
        test_nearest_neighbor_west,
        test_nearest_neighbor_north,
        test_nearest_neighbor_priority,
        test_signal_follow_requires_isolated_defect,
        test_all_16_signal_patterns,

        test_west_correction_edge,
        test_north_correction_edge,

        test_reset,
        test_reset_schedule,
        test_total_runtime,

        test_small_code_capacity,
        test_even_syndrome_parity,
    ]

    print("SCALA2D paper-reference tests")
    print("=" * 44)

    for test in tests:
        test()
        print(f"PASS  {test.__name__}")

    print("=" * 44)
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    run_tests()
