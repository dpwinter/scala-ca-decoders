"""
Equivalence tests for Harrington2D reference vs optimized implementation.

Run:

    python -m scripts.test_harrington2d_optimized
"""

import random

import numpy as np

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
    NEIGHBORS,
    CARDINAL,
    Harrington2D,
    syndrome,
    apply_corrections,
    harrington_rule,
    level_role,
    zeros,
)

from src.harrington2d_optimized import (
    _build_geometry,
    _rule_masks,
    _seed_rng,
    _step,
)


U = 16
FN = 0.4
FC = 0.9


# ============================================================
# Helpers
# ============================================================

def bit(word, lane=0):
    return bool(
        (int(word) >> lane) & 1
    )


def pack_frames(frames):
    """
    frames = [(h0,v0), (h1,v1), ...]
    """

    d = len(frames[0][0])

    h = np.zeros(
        (d, d),
        dtype=np.uint64,
    )

    v = np.zeros(
        (d, d),
        dtype=np.uint64,
    )

    for lane, (hl, vl) in enumerate(frames):

        mask = np.uint64(1 << lane)

        for y in range(d):
            for x in range(d):

                if hl[y][x]:
                    h[y, x] |= mask

                if vl[y][x]:
                    v[y, x] |= mask

    return h, v


def unpack_frame(
    h,
    v,
    lane,
):
    d = h.shape[0]

    hh = zeros(d)
    vv = zeros(d)

    for y in range(d):
        for x in range(d):

            hh[y][x] = bit(
                h[y, x],
                lane,
            )

            vv[y][x] = bit(
                v[y, x],
                lane,
            )

    return hh, vv


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
            value |= (
                1 << b
            )

    return value


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

    count_sig = np.zeros(
        (
            n_levels,
            d,
            d,
            8,
        ),
        dtype=np.uint64,
    )

    flip_sig = np.zeros(
        (
            n_levels,
            d,
            d,
            4,
        ),
        dtype=np.uint64,
    )

    if lanes == 64:
        valid_mask = np.uint64(
            0xFFFFFFFFFFFFFFFF
        )
    else:
        valid_mask = np.uint64(
            (1 << lanes) - 1
        )

    return {
        "geometry": geometry,

        "horizontal": np.zeros(
            (d, d),
            dtype=np.uint64,
        ),

        "vertical": np.zeros(
            (d, d),
            dtype=np.uint64,
        ),

        "defects": np.zeros(
            (d, d),
            dtype=np.uint64,
        ),

        "count_sig": count_sig,
        "flip_sig": flip_sig,

        "new_count_sig": np.zeros_like(
            count_sig
        ),

        "new_flip_sig": np.zeros_like(
            flip_sig
        ),

        "counters": np.zeros(
            (
                9,
                geometry["max_bits"],
                geometry["rep_count"],
            ),
            dtype=np.uint64,
        ),

        "ages": np.zeros(
            n_levels,
            dtype=np.int64,
        ),

        "valid_mask": valid_mask,

        "rng": _seed_rng(
            12345
        ),
    }


def optimized_step(
    state,
    p_signal_threshold=0,
):
    g = state["geometry"]

    _step(
        state["horizontal"],
        state["vertical"],
        state["defects"],

        state["count_sig"],
        state["flip_sig"],

        state["new_count_sig"],
        state["new_flip_sig"],

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
        p_signal_threshold,
        state["rng"],

        state["horizontal"].shape[0],
        g["n_levels"],
    )


# ============================================================
# Geometry
# ============================================================

def test_geometry():
    for d in (
        3,
        9,
        27,
    ):

        geometry = _build_geometry(
            d,
            U,
            FN,
            FC,
        )

        for level in range(
            geometry["n_levels"]
        ):

            k = level + 1

            for y in range(d):
                for x in range(d):

                    expected = level_role(
                        y,
                        x,
                        k,
                    )

                    actual = geometry[
                        "roles"
                    ][
                        level,
                        y,
                        x,
                    ]

                    assert (
                        actual
                        == expected
                    ), (
                        f"d={d}, "
                        f"k={k}, "
                        f"({y},{x}): "
                        f"{actual} != {expected}"
                    )


# ============================================================
# Packed local rule
# ============================================================

def test_local_rule_exhaustive():
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

        for state in range(
            1 << 9
        ):

            defects_ref = [
                bool(
                    (state >> i)
                    & 1
                )
                for i in range(9)
            ]

            expected = harrington_rule(
                role,
                defects_ref,
            )

            defects_opt = np.zeros(
                9,
                dtype=np.uint64,
            )

            for i in range(9):
                defects_opt[i] = np.uint64(
                    int(defects_ref[i])
                )

            (
                move_N,
                move_W,
                move_E,
                move_S,
            ) = _rule_masks(
                role,
                defects_opt,
                np.uint64(1),
            )

            actual = NONE

            if bit(move_N):
                actual = N

            if bit(move_W):
                assert actual == NONE
                actual = W

            if bit(move_E):
                assert actual == NONE
                actual = E

            if bit(move_S):
                assert actual == NONE
                actual = S

            assert (
                actual == expected
            ), (
                f"role={role}, "
                f"state={state}: "
                f"{actual} != {expected}"
            )


# ============================================================
# Reference state comparison
# ============================================================

def assert_internal_state_equal(
    ref,
    opt,
    lane=0,
    label="",
):
    d = ref.d
    g = opt["geometry"]

    for level in range(
        g["n_levels"]
    ):

        # All cells share the same clock at one level.
        expected_age = (
            ref.memory[0][0][level]
            .age
        )

        actual_age = int(
            opt["ages"][level]
        )

        assert (
            actual_age
            == expected_age
        ), (
            f"{label}: "
            f"age mismatch "
            f"k={level + 1}: "
            f"{actual_age} != "
            f"{expected_age}"
        )

        for y in range(d):
            for x in range(d):

                mem = (
                    ref.memory[y][x][level]
                )

                assert (
                    mem.age
                    == expected_age
                )

                # CountSignals
                for direction in NEIGHBORS:

                    actual = bit(
                        opt[
                            "count_sig"
                        ][
                            level,
                            y,
                            x,
                            direction,
                        ],
                        lane,
                    )

                    expected = bool(
                        mem.count_signal[
                            direction
                        ]
                    )

                    assert (
                        actual
                        == expected
                    ), (
                        f"{label}: "
                        f"CountSig mismatch "
                        f"k={level + 1}, "
                        f"({y},{x}), "
                        f"dir={direction}: "
                        f"{actual} != "
                        f"{expected}"
                    )

                # FlipSignals
                for direction in CARDINAL:

                    actual = bit(
                        opt[
                            "flip_sig"
                        ][
                            level,
                            y,
                            x,
                            direction,
                        ],
                        lane,
                    )

                    expected = bool(
                        mem.flip_signal[
                            direction
                        ]
                    )

                    assert (
                        actual
                        == expected
                    ), (
                        f"{label}: "
                        f"FlipSig mismatch "
                        f"k={level + 1}, "
                        f"({y},{x}), "
                        f"dir={direction}: "
                        f"{actual} != "
                        f"{expected}"
                    )

                rep = int(
                    g[
                        "site_to_rep"
                    ][
                        level,
                        y,
                        x,
                    ]
                )

                if rep < 0:
                    continue

                nbits = int(
                    g[
                        "nbits_levels"
                    ][level]
                )

                for counter_type in range(
                    9
                ):

                    actual = (
                        counter_value(
                            opt[
                                "counters"
                            ],
                            counter_type,
                            rep,
                            nbits,
                            lane,
                        )
                    )

                    expected = (
                        mem.count[
                            counter_type
                        ]
                    )

                    assert (
                        actual
                        == expected
                    ), (
                        f"{label}: "
                        f"counter mismatch "
                        f"k={level + 1}, "
                        f"({y},{x}), "
                        f"type="
                        f"{counter_type}: "
                        f"{actual} != "
                        f"{expected}"
                    )


# ============================================================
# One reference round
# ============================================================

def reference_step(
    decoder,
    h,
    v,
):
    defects = syndrome(
        h,
        v,
    )

    corrections = decoder.step(
        defects,
        p_signal=0.0,
    )

    return apply_corrections(
        h,
        v,
        corrections,
    )


# ============================================================
# Single-lane trajectory equivalence
# ============================================================

def test_single_trajectory_equivalence():
    rng = random.Random(
        12345
    )

    cases = [
        (9, 30, 80),
        (27, 6, 80),
    ]

    for d, trials, steps in cases:

        for trial in range(
            trials
        ):

            h_ref = [
                [
                    rng.random()
                    < 0.15
                    for _ in range(d)
                ]
                for _ in range(d)
            ]

            v_ref = [
                [
                    rng.random()
                    < 0.15
                    for _ in range(d)
                ]
                for _ in range(d)
            ]

            decoder_ref = Harrington2D(
                d,
                U=U,
                fN=FN,
                fC=FC,
                seed=1,
            )

            opt = make_optimized_state(
                d,
                lanes=1,
            )

            for y in range(d):
                for x in range(d):

                    opt[
                        "horizontal"
                    ][y, x] = np.uint64(
                        int(h_ref[y][x])
                    )

                    opt[
                        "vertical"
                    ][y, x] = np.uint64(
                        int(v_ref[y][x])
                    )

            for t in range(steps):

                h_ref, v_ref = (
                    reference_step(
                        decoder_ref,
                        h_ref,
                        v_ref,
                    )
                )

                optimized_step(
                    opt
                )

                h_opt, v_opt = (
                    unpack_frame(
                        opt["horizontal"],
                        opt["vertical"],
                        0,
                    )
                )

                assert (
                    h_opt == h_ref
                ), (
                    f"d={d}, "
                    f"trial={trial}, "
                    f"t={t + 1}: "
                    "horizontal frame mismatch"
                )

                assert (
                    v_opt == v_ref
                ), (
                    f"d={d}, "
                    f"trial={trial}, "
                    f"t={t + 1}: "
                    "vertical frame mismatch"
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
# Multispin equivalence
# ============================================================

def test_multispin_equivalence():
    rng = random.Random(
        54321
    )

    d = 9
    lanes = 12
    steps = 80

    frames_ref = []

    decoders = []

    for _ in range(lanes):

        h = [
            [
                rng.random() < 0.20
                for _ in range(d)
            ]
            for _ in range(d)
        ]

        v = [
            [
                rng.random() < 0.20
                for _ in range(d)
            ]
            for _ in range(d)
        ]

        frames_ref.append(
            [h, v]
        )

        decoders.append(
            Harrington2D(
                d,
                U=U,
                fN=FN,
                fC=FC,
                seed=1,
            )
        )

    opt = make_optimized_state(
        d,
        lanes=lanes,
    )

    packed_h, packed_v = (
        pack_frames(
            [
                (frame[0], frame[1])
                for frame in frames_ref
            ]
        )
    )

    opt["horizontal"][:] = packed_h
    opt["vertical"][:] = packed_v

    for t in range(steps):

        for lane in range(
            lanes
        ):

            h_ref, v_ref = (
                reference_step(
                    decoders[lane],
                    frames_ref[lane][0],
                    frames_ref[lane][1],
                )
            )

            frames_ref[lane][0] = h_ref
            frames_ref[lane][1] = v_ref

        optimized_step(opt)

        for lane in range(
            lanes
        ):

            h_opt, v_opt = (
                unpack_frame(
                    opt["horizontal"],
                    opt["vertical"],
                    lane,
                )
            )

            assert (
                h_opt
                == frames_ref[lane][0]
            ), (
                f"lane={lane}, "
                f"t={t + 1}: "
                "horizontal mismatch"
            )

            assert (
                v_opt
                == frames_ref[lane][1]
            ), (
                f"lane={lane}, "
                f"t={t + 1}: "
                "vertical mismatch"
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
# d=3 single-edge trajectories
# ============================================================

def decode_steps_reference(
    h,
    v,
    max_steps=500,
):
    d = len(h)

    decoder = Harrington2D(
        d,
        U=U,
        fN=FN,
        fC=FC,
        seed=1,
    )

    h = [
        row.copy()
        for row in h
    ]

    v = [
        row.copy()
        for row in v
    ]

    for t in range(
        max_steps + 1
    ):

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

        if t == max_steps:
            break

        h, v = reference_step(
            decoder,
            h,
            v,
        )

    return (
        h,
        v,
        False,
        max_steps,
    )


def decode_steps_optimized(
    h,
    v,
    max_steps=500,
):
    d = len(h)

    opt = make_optimized_state(
        d,
        lanes=1,
    )

    for y in range(d):
        for x in range(d):

            opt[
                "horizontal"
            ][y, x] = np.uint64(
                int(h[y][x])
            )

            opt[
                "vertical"
            ][y, x] = np.uint64(
                int(v[y][x])
            )

    for t in range(
        max_steps + 1
    ):

        hh, vv = unpack_frame(
            opt["horizontal"],
            opt["vertical"],
            0,
        )

        defects = syndrome(
            hh,
            vv,
        )

        if not any(
            any(row)
            for row in defects
        ):
            return (
                hh,
                vv,
                True,
                t,
            )

        if t == max_steps:
            break

        optimized_step(
            opt
        )

    hh, vv = unpack_frame(
        opt["horizontal"],
        opt["vertical"],
        0,
    )

    return (
        hh,
        vv,
        False,
        max_steps,
    )


def test_all_single_errors_d3():
    d = 3

    # Horizontal
    for y in range(d):
        for x in range(d):

            h = zeros(d)
            v = zeros(d)

            h[y][x] = True

            ref = decode_steps_reference(
                h,
                v,
            )

            opt = decode_steps_optimized(
                h,
                v,
            )

            assert opt == ref, (
                f"horizontal "
                f"({y},{x}) mismatch"
            )

    # Vertical
    for y in range(d):
        for x in range(d):

            h = zeros(d)
            v = zeros(d)

            v[y][x] = True

            ref = decode_steps_reference(
                h,
                v,
            )

            opt = decode_steps_optimized(
                h,
                v,
            )

            assert opt == ref, (
                f"vertical "
                f"({y},{x}) mismatch"
            )


# ============================================================
# Random d=9 decoder trajectories
# ============================================================

def test_random_d9_decoding():
    rng = random.Random(
        999
    )

    d = 9

    for trial in range(40):

        h = [
            [
                rng.random() < 0.08
                for _ in range(d)
            ]
            for _ in range(d)
        ]

        v = [
            [
                rng.random() < 0.08
                for _ in range(d)
            ]
            for _ in range(d)
        ]

        ref = decode_steps_reference(
            h,
            v,
            max_steps=2000,
        )

        opt = decode_steps_optimized(
            h,
            v,
            max_steps=2000,
        )

        assert opt == ref, (
            f"d=9 trial {trial} mismatch"
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
        test_all_single_errors_d3,
        test_random_d9_decoding,
    ]

    print(
        "Harrington2D optimized/reference equivalence tests"
    )

    print("=" * 62)

    for test in tests:
        test()

        print(
            f"PASS  {test.__name__}"
        )

    print("=" * 62)

    print(
        f"All {len(tests)} "
        "equivalence tests passed."
    )


if __name__ == "__main__":
    run_tests()
