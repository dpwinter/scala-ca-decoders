"""
Exact equivalence tests between

    src/harrington1d.py
    src/harrington1d_optimized.py

Run from the project root with:

    python -m scripts.test_harrington1d_optimized
"""

import random

import numpy as np

from src.harrington1d import (
    LEFT,
    RIGHT,
    CENTER,
    NONE,
    Harrington1D,
    syndrome,
    apply_corrections,
    has_logical_error,
    harrington_rule,
    level_role,
)

from src.harrington1d_optimized import (
    _build_geometry,
    _rule_masks,
    _seed_rng,
    _step,
)


U = 10
FN = 0.4
FC = 0.9

MAX_STEPS = 10_000


# ============================================================
# Helpers
# ============================================================

def bit(word, lane=0):
    return bool(
        (int(word) >> lane) & 1
    )


def counter_value(
    counters,
    counter_type,
    rep,
    nbits,
    lane=0,
):
    value = 0

    for b in range(nbits):
        if bit(
            counters[
                counter_type,
                b,
                rep,
            ],
            lane,
        ):
            value |= 1 << b

    return value


def pack_configurations(configurations):
    d = len(configurations[0])

    packed = np.zeros(
        d,
        dtype=np.uint64,
    )

    for lane, config in enumerate(
        configurations
    ):
        for i, value in enumerate(config):
            if value:
                packed[i] |= np.uint64(
                    1 << lane
                )

    return packed


def unpack_configuration(
    packed,
    lane,
):
    return [
        bit(packed[i], lane)
        for i in range(len(packed))
    ]


# ============================================================
# Optimized state
# ============================================================

def make_optimized_state(
    d,
    lanes=1,
):
    geometry = _build_geometry(
        d,
        U,
        FN,
        FC,
    )

    n_levels = geometry[
        "n_levels"
    ]

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

    state = {
        "geometry": geometry,

        "qubits": np.zeros(
            d,
            dtype=np.uint64,
        ),

        "defects": np.zeros(
            d,
            dtype=np.uint64,
        ),

        "count_L": count_L,
        "count_R": count_R,
        "flip_L": flip_L,
        "flip_R": flip_R,

        "new_count_L": np.zeros_like(
            count_L
        ),
        "new_count_R": np.zeros_like(
            count_L
        ),
        "new_flip_L": np.zeros_like(
            count_L
        ),
        "new_flip_R": np.zeros_like(
            count_L
        ),

        "counters": np.zeros(
            (
                3,
                geometry["max_bits"],
                geometry["rep_count"],
            ),
            dtype=np.uint64,
        ),

        "ages": np.zeros(
            n_levels,
            dtype=np.int64,
        ),

        "valid_mask": np.uint64(
            (1 << lanes) - 1
        )
        if lanes < 64
        else np.uint64(
            0xFFFFFFFFFFFFFFFF
        ),

        "rng": _seed_rng(12345),
    }

    return state


def optimized_step(state):
    g = state["geometry"]

    _step(
        state["qubits"],
        state["defects"],

        state["count_L"],
        state["count_R"],
        state["flip_L"],
        state["flip_R"],

        state["new_count_L"],
        state["new_count_R"],
        state["new_flip_L"],
        state["new_flip_R"],

        state["counters"],

        g["site_to_rep"],
        g["roles"],
        state["ages"],
        g["U_levels"],
        g["Q_levels"],
        g["periods"],
        g["nbits_levels"],
        g["threshold_N"],
        g["threshold_C"],

        state["valid_mask"],

        0,  # no signal noise

        state["rng"],

        len(state["qubits"]),
        g["n_levels"],
    )


# ============================================================
# Geometry
# ============================================================

def test_geometry():
    for d in (3, 9, 27, 81):

        geometry = _build_geometry(
            d,
            U,
            FN,
            FC,
        )

        depth = round(
            np.log(d)
            / np.log(3)
        )

        assert (
            geometry["n_levels"]
            == depth - 1
        )

        for level in range(
            geometry["n_levels"]
        ):
            k = level + 1

            for i in range(d):

                reference_role = (
                    level_role(i, k)
                )

                optimized_role = (
                    geometry[
                        "roles"
                    ][level, i]
                )

                assert (
                    optimized_role
                    == reference_role
                ), (
                    f"d={d}, "
                    f"k={k}, "
                    f"i={i}: "
                    f"reference="
                    f"{reference_role}, "
                    f"optimized="
                    f"{optimized_role}"
                )


# ============================================================
# Packed local rule
# ============================================================

def test_local_rule_exhaustive():
    """
    Check all local roles and all 2^3 defect patterns.
    """

    for role in (
        LEFT,
        CENTER,
        RIGHT,
        NONE,
    ):
        for left in (
            False,
            True,
        ):
            for center in (
                False,
                True,
            ):
                for right in (
                    False,
                    True,
                ):

                    expected = (
                        harrington_rule(
                            role,
                            left,
                            center,
                            right,
                        )
                    )

                    move_L, move_R = (
                        _rule_masks(
                            role,
                            np.uint64(left),
                            np.uint64(center),
                            np.uint64(right),
                            np.uint64(1),
                        )
                    )

                    actual = NONE

                    if bit(move_L):
                        actual = LEFT

                    if bit(move_R):
                        assert (
                            actual == NONE
                        )

                        actual = RIGHT

                    assert actual == expected, (
                        f"role={role}, "
                        f"(l,c,r)="
                        f"({left},{center},{right}), "
                        f"reference={expected}, "
                        f"optimized={actual}"
                    )


# ============================================================
# Compare complete internal state
# ============================================================

def assert_internal_state_equal(
    reference,
    optimized,
    lane=0,
    label="",
):
    d = reference.d

    g = optimized[
        "geometry"
    ]

    # --------------------------------------------------------
    # Ages
    # --------------------------------------------------------

    for level in range(
        g["n_levels"]
    ):
        expected_age = (
            reference
            .memory[0][level]
            .age
        )

        actual_age = int(
            optimized[
                "ages"
            ][level]
        )

        assert (
            actual_age
            == expected_age
        ), (
            f"{label}: "
            f"level {level + 1} age "
            f"{actual_age} != "
            f"{expected_age}"
        )

        for i in range(d):
            assert (
                reference
                .memory[i][level]
                .age
                == expected_age
            )

    # --------------------------------------------------------
    # Signals and counters
    # --------------------------------------------------------

    for level in range(
        g["n_levels"]
    ):

        nbits = int(
            g["nbits_levels"][
                level
            ]
        )

        for i in range(d):

            mem = (
                reference
                .memory[i][level]
            )

            assert (
                bit(
                    optimized[
                        "count_L"
                    ][level, i],
                    lane,
                )
                == bool(
                    mem.count_signal[
                        LEFT
                    ]
                )
            ), (
                f"{label}: "
                f"CountSig L mismatch "
                f"k={level + 1}, i={i}"
            )

            assert (
                bit(
                    optimized[
                        "count_R"
                    ][level, i],
                    lane,
                )
                == bool(
                    mem.count_signal[
                        RIGHT
                    ]
                )
            ), (
                f"{label}: "
                f"CountSig R mismatch "
                f"k={level + 1}, i={i}"
            )

            assert (
                bit(
                    optimized[
                        "flip_L"
                    ][level, i],
                    lane,
                )
                == bool(
                    mem.flip_signal[
                        LEFT
                    ]
                )
            ), (
                f"{label}: "
                f"FlipSig L mismatch "
                f"k={level + 1}, i={i}"
            )

            assert (
                bit(
                    optimized[
                        "flip_R"
                    ][level, i],
                    lane,
                )
                == bool(
                    mem.flip_signal[
                        RIGHT
                    ]
                )
            ), (
                f"{label}: "
                f"FlipSig R mismatch "
                f"k={level + 1}, i={i}"
            )

            rep = int(
                g["site_to_rep"][
                    level,
                    i,
                ]
            )

            if rep < 0:
                continue

            for counter_type in (
                LEFT,
                RIGHT,
                CENTER,
            ):
                actual = counter_value(
                    optimized[
                        "counters"
                    ],
                    counter_type,
                    rep,
                    nbits,
                    lane,
                )

                expected = (
                    mem.count[
                        counter_type
                    ]
                )

                assert (
                    actual == expected
                ), (
                    f"{label}: "
                    f"counter mismatch "
                    f"k={level + 1}, "
                    f"i={i}, "
                    f"type={counter_type}: "
                    f"{actual} != "
                    f"{expected}"
                )


# ============================================================
# Single-lane trajectory equivalence
# ============================================================

def test_single_trajectory_equivalence():
    rng = random.Random(
        12345
    )

    for d in (9, 27):

        trials = (
            40
            if d == 9
            else 15
        )

        steps = (
            100
            if d == 9
            else 150
        )

        for trial in range(trials):

            qubits_ref = [
                rng.random() < 0.25
                for _ in range(d)
            ]

            decoder_ref = (
                Harrington1D(
                    d=d,
                    U=U,
                    fN=FN,
                    fC=FC,
                )
            )

            opt = (
                make_optimized_state(
                    d,
                    lanes=1,
                )
            )

            opt["qubits"][:] = np.array(
                qubits_ref,
                dtype=np.uint64,
            )

            for t in range(steps):

                defects = syndrome(
                    qubits_ref
                )

                corrections = (
                    decoder_ref.step(
                        defects
                    )
                )

                qubits_ref = (
                    apply_corrections(
                        qubits_ref,
                        corrections,
                    )
                )

                optimized_step(
                    opt
                )

                qubits_opt = (
                    unpack_configuration(
                        opt["qubits"],
                        0,
                    )
                )

                assert (
                    qubits_opt
                    == qubits_ref
                ), (
                    f"d={d}, "
                    f"trial={trial}, "
                    f"t={t + 1}\n"
                    f"reference={qubits_ref}\n"
                    f"optimized={qubits_opt}"
                )

                assert_internal_state_equal(
                    decoder_ref,
                    opt,
                    lane=0,
                    label=(
                        f"d={d}, "
                        f"trial={trial}, "
                        f"t={t + 1}"
                    ),
                )


# ============================================================
# Multispin trajectory equivalence
# ============================================================

def test_multispin_equivalence():
    """
    Run several independent reference decoders while packing
    them into different bits of one optimized uint64 state.
    """

    rng = random.Random(
        54321
    )

    d = 9
    lanes = 16
    steps = 100

    qubits_ref = [
        [
            rng.random() < 0.30
            for _ in range(d)
        ]
        for _ in range(lanes)
    ]

    decoders = [
        Harrington1D(
            d=d,
            U=U,
            fN=FN,
            fC=FC,
        )
        for _ in range(lanes)
    ]

    opt = make_optimized_state(
        d,
        lanes=lanes,
    )

    opt["qubits"][:] = (
        pack_configurations(
            qubits_ref
        )
    )

    for t in range(steps):

        for lane in range(lanes):

            defects = syndrome(
                qubits_ref[lane]
            )

            corrections = (
                decoders[lane].step(
                    defects
                )
            )

            qubits_ref[lane] = (
                apply_corrections(
                    qubits_ref[lane],
                    corrections,
                )
            )

        optimized_step(opt)

        for lane in range(lanes):

            actual = (
                unpack_configuration(
                    opt["qubits"],
                    lane,
                )
            )

            assert (
                actual
                == qubits_ref[lane]
            ), (
                f"lane={lane}, "
                f"t={t + 1}\n"
                f"reference="
                f"{qubits_ref[lane]}\n"
                f"optimized="
                f"{actual}"
            )

            assert_internal_state_equal(
                decoders[lane],
                opt,
                lane=lane,
                label=(
                    f"lane={lane}, "
                    f"t={t + 1}"
                ),
            )


# ============================================================
# Full deterministic decoder helpers
# ============================================================

def decode_reference(
    initial,
):
    qubits = initial.copy()

    decoder = Harrington1D(
        d=len(qubits),
        U=U,
        fN=FN,
        fC=FC,
    )

    for t in range(
        MAX_STEPS
    ):
        defects = syndrome(
            qubits
        )

        if not any(defects):
            return (
                qubits,
                True,
                t,
            )

        corrections = (
            decoder.step(
                defects
            )
        )

        qubits = (
            apply_corrections(
                qubits,
                corrections,
            )
        )

    return (
        qubits,
        False,
        MAX_STEPS,
    )


def decode_optimized(
    initial,
):
    d = len(initial)

    opt = make_optimized_state(
        d,
        lanes=1,
    )

    opt["qubits"][:] = np.array(
        initial,
        dtype=np.uint64,
    )

    for t in range(
        MAX_STEPS
    ):
        current = (
            unpack_configuration(
                opt["qubits"],
                0,
            )
        )

        if not any(
            syndrome(current)
        ):
            return (
                current,
                True,
                t,
            )

        optimized_step(
            opt
        )

    return (
        unpack_configuration(
            opt["qubits"],
            0,
        ),
        False,
        MAX_STEPS,
    )


# ============================================================
# Exhaustive d=9 decoder equivalence
# ============================================================

def test_all_d9_configurations():
    """
    Compare all 2^9 = 512 initial physical configurations.
    """

    d = 9

    for state in range(
        1 << d
    ):
        initial = [
            bool(
                (state >> i) & 1
            )
            for i in range(d)
        ]

        (
            ref_final,
            ref_resolved,
            ref_steps,
        ) = decode_reference(
            initial
        )

        (
            opt_final,
            opt_resolved,
            opt_steps,
        ) = decode_optimized(
            initial
        )

        assert (
            opt_resolved
            == ref_resolved
        ), (
            f"state={state}: "
            "resolved mismatch"
        )

        assert (
            opt_steps
            == ref_steps
        ), (
            f"state={state}: "
            f"step mismatch "
            f"{opt_steps} != "
            f"{ref_steps}"
        )

        assert (
            opt_final
            == ref_final
        ), (
            f"state={state}\n"
            f"initial={initial}\n"
            f"reference={ref_final}\n"
            f"optimized={opt_final}"
        )

        assert (
            has_logical_error(
                opt_final
            )
            == has_logical_error(
                ref_final
            )
        )


# ============================================================
# Run
# ============================================================

def run_tests():
    tests = [
        test_geometry,
        test_local_rule_exhaustive,
        test_single_trajectory_equivalence,
        test_multispin_equivalence,
        test_all_d9_configurations,
    ]

    print(
        "Harrington1D optimized/reference "
        "equivalence tests"
    )

    print("=" * 58)

    for test in tests:
        test()

        print(
            f"PASS  {test.__name__}"
        )

    print("=" * 58)

    print(
        f"All {len(tests)} "
        "equivalence tests passed."
    )


if __name__ == "__main__":
    run_tests()
