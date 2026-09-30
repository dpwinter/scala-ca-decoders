"""
Tests for the minimal Harrington1D reference implementation.

The key code-capacity check is exact:

Harrington1D should reproduce the concatenated-majority
logical-error-rate polynomial for d = 3^m.

For d=3 and d=9 we exhaust all physical configurations and
compare the exact number of failures at each physical weight.

Run from the project root with:

    python -m scripts.test_harrington1d
"""

from itertools import product

from src.harrington1d import (
    LEFT,
    RIGHT,
    CENTER,
    NONE,
    Memory,
    Harrington1D,
    syndrome,
    has_logical_error,
    apply_corrections,
    harrington_rule,
    level_role,
    hierarchy_depth,
    decode_code_capacity,
)


# ============================================================
# Parameters
# ============================================================

U = 10
FN = 0.4
FC = 0.9

# Generous runtime for correctness tests.
MAX_STEPS = 10_000


# ============================================================
# Concatenated majority decoder
# ============================================================

def majority3(bits):
    assert len(bits) == 3
    return sum(bits) >= 2


def concatenated_majority(bits):
    bits = list(bits)

    while len(bits) > 1:
        assert len(bits) % 3 == 0

        bits = [
            majority3(bits[i:i + 3])
            for i in range(0, len(bits), 3)
        ]

    return bits[0]


def majority_map(p):
    return 3 * p**2 - 2 * p**3


def concatenated_logical_rate(p, d):
    depth = hierarchy_depth(d)

    pL = p

    for _ in range(depth):
        pL = majority_map(pL)

    return pL


# ============================================================
# Repetition-code geometry
# ============================================================

def test_syndrome():
    qubits = [
        False,
        True,
        True,
        False,
        False,
    ]

    assert syndrome(qubits) == [
        True,
        False,
        True,
        False,
        False,
    ]


def test_periodic_syndrome():
    qubits = [
        True,
        False,
        False,
        False,
        False,
    ]

    defects = syndrome(qubits)

    assert defects == [
        True,
        False,
        False,
        False,
        True,
    ]


def test_logical_error():
    assert not has_logical_error(
        [False, False, True]
    )

    assert has_logical_error(
        [True, True, False]
    )

    assert not has_logical_error(
        [True, True, False, False, False]
    )

    assert has_logical_error(
        [True, True, True, False, False]
    )


# ============================================================
# Physical correction convention
# ============================================================

def test_apply_left_correction():
    qubits = [
        False,
        False,
        False,
    ]

    corrections = [
        LEFT,
        NONE,
        NONE,
    ]

    result = apply_corrections(
        qubits,
        corrections,
    )

    assert result == [
        True,
        False,
        False,
    ]


def test_apply_right_correction():
    qubits = [
        False,
        False,
        False,
    ]

    corrections = [
        RIGHT,
        NONE,
        NONE,
    ]

    result = apply_corrections(
        qubits,
        corrections,
    )

    assert result == [
        False,
        True,
        False,
    ]


def test_correction_cancellation():
    qubits = [
        False,
        False,
        False,
    ]

    corrections = [
        RIGHT,   # cell 0 -> qubit 1
        LEFT,    # cell 1 -> qubit 1
        NONE,
    ]

    result = apply_corrections(
        qubits,
        corrections,
    )

    assert result == qubits


# ============================================================
# Local Harrington rule
# ============================================================

def test_local_rule_center():
    for left in (False, True):
        for center in (False, True):
            for right in (False, True):

                assert harrington_rule(
                    CENTER,
                    left,
                    center,
                    right,
                ) == NONE


def test_local_rule_no_defect():
    for role in (LEFT, RIGHT):

        assert harrington_rule(
            role,
            True,
            False,
            True,
        ) == NONE


def test_local_rule_left_cell():
    assert harrington_rule(
        LEFT,
        True,
        True,
        False,
    ) == LEFT

    assert harrington_rule(
        LEFT,
        False,
        True,
        False,
    ) == RIGHT


def test_local_rule_right_cell():
    assert harrington_rule(
        RIGHT,
        False,
        True,
        True,
    ) == NONE

    assert harrington_rule(
        RIGHT,
        False,
        True,
        False,
    ) == LEFT


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
        hierarchy_depth(10)

    except ValueError:
        return

    raise AssertionError(
        "non-power-of-three distance was accepted"
    )


def test_level_zero_roles():
    expected = [
        LEFT,
        CENTER,
        RIGHT,
        LEFT,
        CENTER,
        RIGHT,
        LEFT,
        CENTER,
        RIGHT,
    ]

    actual = [
        level_role(i, 0)
        for i in range(9)
    ]

    assert actual == expected


def test_level_one_centers_d9():
    roles = [
        level_role(i, 1)
        for i in range(9)
    ]

    expected = [
        NONE,
        LEFT,
        NONE,
        NONE,
        CENTER,
        NONE,
        NONE,
        RIGHT,
        NONE,
    ]

    assert roles == expected


# ============================================================
# Memory timing
# ============================================================

def test_memory_parameters_d9():
    decoder = Harrington1D(
        d=9,
        U=U,
        fN=FN,
        fC=FC,
    )

    for i in range(9):
        assert len(decoder.memory[i]) == 1

        mem = decoder.memory[i][0]

        assert mem.U == U
        assert mem.Q == 3


def test_memory_clock():
    mem = Memory(
        role=LEFT,
        U=4,
        Q=3,
    )

    period = mem.U + mem.Q

    ages = []

    for _ in range(period):
        mem.age = (
            mem.age + 1
        ) % period

        ages.append(mem.age)

    assert ages == [
        1,
        2,
        3,
        4,
        5,
        6,
        0,
    ]


# ============================================================
# Exact code-capacity performance
# ============================================================

def failure_counts_harrington(d):
    """
    Exhaustively count Harrington failures at every
    physical error weight.
    """

    failures = [
        0 for _ in range(d + 1)
    ]

    unresolved = [
        0 for _ in range(d + 1)
    ]

    for bits in product(
        (False, True),
        repeat=d,
    ):
        qubits = list(bits)
        weight = sum(bits)

        final, resolved, _ = (
            decode_code_capacity(
                qubits,
                U=U,
                fN=FN,
                fC=FC,
                max_steps=MAX_STEPS,
            )
        )

        if not resolved:
            unresolved[weight] += 1
            failures[weight] += 1

        elif has_logical_error(final):
            failures[weight] += 1

    return failures, unresolved


def failure_counts_majority(d):
    """
    Exact failure counts of recursive concatenated majority.
    """

    failures = [
        0 for _ in range(d + 1)
    ]

    for bits in product(
        (False, True),
        repeat=d,
    ):
        weight = sum(bits)

        if concatenated_majority(bits):
            failures[weight] += 1

    return failures


def test_d3_concatenated_performance():
    harrington, unresolved = (
        failure_counts_harrington(3)
    )

    majority = (
        failure_counts_majority(3)
    )

    assert unresolved == [
        0,
        0,
        0,
        0,
    ]

    assert harrington == majority

    assert majority == [
        0,
        0,
        3,
        1,
    ]


def test_d9_concatenated_performance():
    """
    Crucial test:

    Harrington and concatenated majority need not fail on
    exactly the same physical configurations.

    They must have the same number of failures at every
    physical error weight, which implies identical p_L(p).
    """

    harrington, unresolved = (
        failure_counts_harrington(9)
    )

    majority = (
        failure_counts_majority(9)
    )

    assert unresolved == [
        0 for _ in range(10)
    ], (
        f"unresolved counts = {unresolved}"
    )

    assert harrington == majority, (
        "\n"
        f"Harrington failures = {harrington}\n"
        f"Majority failures   = {majority}"
    )

    assert majority == [
        0,
        0,
        0,
        0,
        27,
        99,
        84,
        36,
        9,
        1,
    ]


# ============================================================
# Exact logical-rate polynomial
# ============================================================

def exact_rate_from_weights(
    p,
    failures,
    d,
):
    pL = 0.0

    for weight, count in enumerate(
        failures
    ):
        pL += (
            count
            * p**weight
            * (1.0 - p)**(d - weight)
        )

    return pL


def test_concatenated_analytical_curve_d3():
    failures = (
        failure_counts_majority(3)
    )

    for p in (
        0.01,
        0.05,
        0.10,
        0.20,
        0.40,
    ):
        exact = exact_rate_from_weights(
            p,
            failures,
            3,
        )

        analytic = (
            concatenated_logical_rate(
                p,
                3,
            )
        )

        assert abs(
            exact - analytic
        ) < 1e-14


def test_concatenated_analytical_curve_d9():
    failures = (
        failure_counts_majority(9)
    )

    for p in (
        0.01,
        0.05,
        0.10,
        0.20,
        0.40,
    ):
        exact = exact_rate_from_weights(
            p,
            failures,
            9,
        )

        analytic = (
            concatenated_logical_rate(
                p,
                9,
            )
        )

        assert abs(
            exact - analytic
        ) < 1e-14


def test_harrington_analytical_curve_d9():
    """
    Directly verify that the exhaustive Harrington weight
    enumerator reproduces the concatenated analytical curve.
    """

    failures, unresolved = (
        failure_counts_harrington(9)
    )

    assert sum(unresolved) == 0

    for p in (
        0.01,
        0.05,
        0.10,
        0.20,
        0.40,
    ):
        exact = exact_rate_from_weights(
            p,
            failures,
            9,
        )

        analytic = (
            concatenated_logical_rate(
                p,
                9,
            )
        )

        assert abs(
            exact - analytic
        ) < 1e-14


# ============================================================
# Run
# ============================================================

def run_tests():
    tests = [
        test_syndrome,
        test_periodic_syndrome,
        test_logical_error,

        test_apply_left_correction,
        test_apply_right_correction,
        test_correction_cancellation,

        test_local_rule_center,
        test_local_rule_no_defect,
        test_local_rule_left_cell,
        test_local_rule_right_cell,

        test_hierarchy_depth,
        test_non_power_of_three_rejected,
        test_level_zero_roles,
        test_level_one_centers_d9,

        test_memory_parameters_d9,
        test_memory_clock,

        test_d3_concatenated_performance,
        test_d9_concatenated_performance,

        test_concatenated_analytical_curve_d3,
        test_concatenated_analytical_curve_d9,
        test_harrington_analytical_curve_d9,
    ]

    print(
        "Harrington1D reference tests"
    )

    print("=" * 48)

    for test in tests:
        test()

        print(
            f"PASS  {test.__name__}"
        )

    print("=" * 48)

    print(
        f"All {len(tests)} tests passed."
    )


if __name__ == "__main__":
    run_tests()
