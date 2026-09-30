"""
Equivalence tests between

    src.scala2d
and
    src.scala2d_optimized

The reference implementation is deliberately simple Python.
The optimized implementation uses packed uint64 + Numba.

These tests verify that the optimized implementation is an exact
acceleration of the reference dynamics.

Run from project root:

    python -m scripts.test_scala2d_optimized
"""

import random

import numpy as np

from src import scala2d as ref
from src import scala2d_optimized as opt


# =============================================================================
# Conversion helpers
# =============================================================================

def bool_grid_to_packed(a):
    """
    Put one ordinary Boolean configuration into lane 0 of a uint64 array.
    """
    return np.asarray(
        a,
        dtype=np.uint64,
    )


def packed_to_bool_grid(a):
    """
    Extract lane 0 from a packed uint64 array.
    """
    return (
        (np.asarray(a, dtype=np.uint64) & np.uint64(1))
        != 0
    ).tolist()


def random_grid(d, p, rng):
    return [
        [
            rng.random() < p
            for _ in range(d)
        ]
        for _ in range(d)
    ]


def empty_grid(d):
    return [
        [False] * d
        for _ in range(d)
    ]


def grids_equal(a, b):
    return all(
        a[y][x] == b[y][x]
        for y in range(len(a))
        for x in range(len(a))
    )


def assert_grid_equal(a, b, name):
    if grids_equal(a, b):
        return

    d = len(a)

    mismatches = []

    for y in range(d):
        for x in range(d):
            if bool(a[y][x]) != bool(b[y][x]):
                mismatches.append(
                    (
                        y,
                        x,
                        bool(a[y][x]),
                        bool(b[y][x]),
                    )
                )

    preview = mismatches[:20]

    raise AssertionError(
        f"{name} mismatch: "
        f"{len(mismatches)} cells differ.\n"
        f"First mismatches:\n{preview}"
    )


# =============================================================================
# Optimized single-step wrapper
# =============================================================================

def optimized_step(
    horizontal,
    vertical,
    north,
    west,
    east,
    south,
):
    """
    Run exactly one optimized SCALA step on lane 0.
    """

    d = len(horizontal)

    h = bool_grid_to_packed(horizontal)
    v = bool_grid_to_packed(vertical)

    n = bool_grid_to_packed(north)
    w = bool_grid_to_packed(west)
    e = bool_grid_to_packed(east)
    s = bool_grid_to_packed(south)

    defect = np.zeros(
        (d, d),
        dtype=np.uint64,
    )

    new_n = np.zeros(
        (d, d),
        dtype=np.uint64,
    )

    new_w = np.zeros(
        (d, d),
        dtype=np.uint64,
    )

    new_e = np.zeros(
        (d, d),
        dtype=np.uint64,
    )

    new_s = np.zeros(
        (d, d),
        dtype=np.uint64,
    )

    opt._step(
        h,
        v,
        n,
        w,
        e,
        s,
        defect,
        new_n,
        new_w,
        new_e,
        new_s,
        d,
    )

    return (
        packed_to_bool_grid(h),
        packed_to_bool_grid(v),
        packed_to_bool_grid(n),
        packed_to_bool_grid(w),
        packed_to_bool_grid(e),
        packed_to_bool_grid(s),
    )


# =============================================================================
# Optimized complete-decoder wrapper
# =============================================================================

def optimized_decode(
    horizontal,
    vertical,
):
    """
    Run the complete optimized paper ramp-up/down decoder on lane 0.
    """

    d = len(horizontal)

    h = bool_grid_to_packed(horizontal)
    v = bool_grid_to_packed(vertical)

    n = np.zeros(
        (d, d),
        dtype=np.uint64,
    )

    w = np.zeros(
        (d, d),
        dtype=np.uint64,
    )

    e = np.zeros(
        (d, d),
        dtype=np.uint64,
    )

    s = np.zeros(
        (d, d),
        dtype=np.uint64,
    )

    defect = np.zeros(
        (d, d),
        dtype=np.uint64,
    )

    new_n = np.zeros(
        (d, d),
        dtype=np.uint64,
    )

    new_w = np.zeros(
        (d, d),
        dtype=np.uint64,
    )

    new_e = np.zeros(
        (d, d),
        dtype=np.uint64,
    )

    new_s = np.zeros(
        (d, d),
        dtype=np.uint64,
    )

    opt._decode_paper(
        h,
        v,
        n,
        w,
        e,
        s,
        defect,
        new_n,
        new_w,
        new_e,
        new_s,
        d,
    )

    return (
        packed_to_bool_grid(h),
        packed_to_bool_grid(v),
    )


# =============================================================================
# Schedule
# =============================================================================

def test_reset_schedule_equivalence():
    for d in range(3, 20, 2):

        ref_schedule = ref.reset_schedule(d)
        opt_schedule = opt.paper_reset_schedule(d)

        assert ref_schedule == opt_schedule

        assert sum(opt_schedule) == d * d


# =============================================================================
# Syndrome
# =============================================================================

def test_syndrome_equivalence():
    rng = random.Random(12345)

    for d in (3, 5, 7, 9):

        for _ in range(100):

            h = random_grid(
                d,
                0.35,
                rng,
            )

            v = random_grid(
                d,
                0.35,
                rng,
            )

            s_ref = ref.syndrome(
                h,
                v,
            )

            s_opt = opt.syndrome_array(
                np.asarray(
                    h,
                    dtype=np.uint8,
                ),
                np.asarray(
                    v,
                    dtype=np.uint8,
                ),
            ).astype(bool).tolist()

            assert_grid_equal(
                s_ref,
                s_opt,
                f"syndrome d={d}",
            )


# =============================================================================
# Hand-picked one-step cases
# =============================================================================

def test_single_step_simple_cases():

    for d in (3, 5, 7):

        # -------------------------------------------------------------
        # Empty configuration
        # -------------------------------------------------------------

        h = empty_grid(d)
        v = empty_grid(d)

        n = empty_grid(d)
        w = empty_grid(d)
        e = empty_grid(d)
        s = empty_grid(d)

        ref_result = ref.step(
            h,
            v,
            n,
            w,
            e,
            s,
        )

        opt_result = optimized_step(
            h,
            v,
            n,
            w,
            e,
            s,
        )

        names = [
            "horizontal",
            "vertical",
            "north",
            "west",
            "east",
            "south",
        ]

        for name, a, b in zip(
            names,
            ref_result,
            opt_result,
        ):
            assert_grid_equal(
                a,
                b,
                f"{name}, empty, d={d}",
            )

        # -------------------------------------------------------------
        # One horizontal error
        # -------------------------------------------------------------

        h = empty_grid(d)
        v = empty_grid(d)

        h[1 % d][1 % d] = True

        n = empty_grid(d)
        w = empty_grid(d)
        e = empty_grid(d)
        s = empty_grid(d)

        ref_result = ref.step(
            h,
            v,
            n,
            w,
            e,
            s,
        )

        opt_result = optimized_step(
            h,
            v,
            n,
            w,
            e,
            s,
        )

        for name, a, b in zip(
            names,
            ref_result,
            opt_result,
        ):
            assert_grid_equal(
                a,
                b,
                f"{name}, horizontal error, d={d}",
            )

        # -------------------------------------------------------------
        # Signals crossing / reflection
        # -------------------------------------------------------------

        h = empty_grid(d)
        v = empty_grid(d)

        n = empty_grid(d)
        w = empty_grid(d)
        e = empty_grid(d)
        s = empty_grid(d)

        e[1 % d][0] = True
        s[0][1 % d] = True

        ref_result = ref.step(
            h,
            v,
            n,
            w,
            e,
            s,
        )

        opt_result = optimized_step(
            h,
            v,
            n,
            w,
            e,
            s,
        )

        for name, a, b in zip(
            names,
            ref_result,
            opt_result,
        ):
            assert_grid_equal(
                a,
                b,
                f"{name}, reflection, d={d}",
            )


# =============================================================================
# Random one-step equivalence
# =============================================================================

def test_random_one_step_equivalence():
    """
    This is the most important local test.

    Randomize BOTH data errors and the full signal state, then require
    every output bit after one synchronous step to agree exactly.
    """

    rng = random.Random(20260915)

    tests_per_d = 1000

    for d in (3, 5, 7):

        for trial in range(tests_per_d):

            # Data configurations.
            h = random_grid(
                d,
                0.30,
                rng,
            )

            v = random_grid(
                d,
                0.30,
                rng,
            )

            # Arbitrary internal signal state.
            #
            # We deliberately do not restrict this to states known to be
            # reachable.  Exact equivalence should hold algebraically for
            # every Boolean local state.
            n = random_grid(
                d,
                0.25,
                rng,
            )

            w = random_grid(
                d,
                0.25,
                rng,
            )

            e = random_grid(
                d,
                0.25,
                rng,
            )

            s = random_grid(
                d,
                0.25,
                rng,
            )

            ref_result = ref.step(
                h,
                v,
                n,
                w,
                e,
                s,
            )

            opt_result = optimized_step(
                h,
                v,
                n,
                w,
                e,
                s,
            )

            names = [
                "horizontal",
                "vertical",
                "north",
                "west",
                "east",
                "south",
            ]

            for name, a, b in zip(
                names,
                ref_result,
                opt_result,
            ):

                if not grids_equal(
                    a,
                    b,
                ):
                    raise AssertionError(
                        "\n"
                        "Random one-step equivalence failed\n"
                        f"d       = {d}\n"
                        f"trial   = {trial}\n"
                        f"field   = {name}\n"
                    )


# =============================================================================
# Packed-lane independence
# =============================================================================

def test_multispin_lane_independence():
    """
    Verify that multiple packed trajectories evolve exactly like
    independent reference simulations.

    This specifically tests the uint64 multispin machinery, rather than
    merely testing lane 0.
    """

    rng = random.Random(13579)

    d = 5
    lanes = 16

    h_states = []
    v_states = []
    n_states = []
    w_states = []
    e_states = []
    s_states = []

    for _ in range(lanes):

        h_states.append(
            random_grid(
                d,
                0.30,
                rng,
            )
        )

        v_states.append(
            random_grid(
                d,
                0.30,
                rng,
            )
        )

        n_states.append(
            random_grid(
                d,
                0.20,
                rng,
            )
        )

        w_states.append(
            random_grid(
                d,
                0.20,
                rng,
            )
        )

        e_states.append(
            random_grid(
                d,
                0.20,
                rng,
            )
        )

        s_states.append(
            random_grid(
                d,
                0.20,
                rng,
            )
        )

    def pack(states):
        out = np.zeros(
            (d, d),
            dtype=np.uint64,
        )

        for lane, state in enumerate(states):

            bit = (
                np.uint64(1)
                << np.uint64(lane)
            )

            for y in range(d):
                for x in range(d):
                    if state[y][x]:
                        out[y, x] |= bit

        return out

    h = pack(h_states)
    v = pack(v_states)

    n = pack(n_states)
    w = pack(w_states)
    e = pack(e_states)
    s = pack(s_states)

    defect = np.zeros(
        (d, d),
        dtype=np.uint64,
    )

    new_n = np.zeros(
        (d, d),
        dtype=np.uint64,
    )

    new_w = np.zeros(
        (d, d),
        dtype=np.uint64,
    )

    new_e = np.zeros(
        (d, d),
        dtype=np.uint64,
    )

    new_s = np.zeros(
        (d, d),
        dtype=np.uint64,
    )

    opt._step(
        h,
        v,
        n,
        w,
        e,
        s,
        defect,
        new_n,
        new_w,
        new_e,
        new_s,
        d,
    )

    packed_outputs = (
        h,
        v,
        n,
        w,
        e,
        s,
    )

    for lane in range(lanes):

        reference = ref.step(
            h_states[lane],
            v_states[lane],
            n_states[lane],
            w_states[lane],
            e_states[lane],
            s_states[lane],
        )

        optimized = tuple(
            opt.extract_lane(
                field,
                lane,
            ).astype(bool).tolist()
            for field in packed_outputs
        )

        names = [
            "horizontal",
            "vertical",
            "north",
            "west",
            "east",
            "south",
        ]

        for name, a, b in zip(
            names,
            reference,
            optimized,
        ):
            assert_grid_equal(
                a,
                b,
                (
                    f"packed lane={lane}, "
                    f"field={name}"
                ),
            )


# =============================================================================
# Complete decoder: exhaustive low-weight errors
# =============================================================================

def test_full_decoder_weight_one():
    """
    Every weight-0 and weight-1 error for d=3 and d=5 must produce
    exactly the same final h/v arrays.
    """

    for d in (3, 5):

        # Empty state.
        h = empty_grid(d)
        v = empty_grid(d)

        ref_h, ref_v = (
            ref.decode_code_capacity(
                h,
                v,
            )
        )

        opt_h, opt_v = (
            optimized_decode(
                h,
                v,
            )
        )

        assert_grid_equal(
            ref_h,
            opt_h,
            f"full decoder h, empty, d={d}",
        )

        assert_grid_equal(
            ref_v,
            opt_v,
            f"full decoder v, empty, d={d}",
        )

        # Every horizontal weight-one error.
        for y in range(d):
            for x in range(d):

                h = empty_grid(d)
                v = empty_grid(d)

                h[y][x] = True

                ref_h, ref_v = (
                    ref.decode_code_capacity(
                        h,
                        v,
                    )
                )

                opt_h, opt_v = (
                    optimized_decode(
                        h,
                        v,
                    )
                )

                assert_grid_equal(
                    ref_h,
                    opt_h,
                    (
                        "full decoder h, "
                        f"d={d}, H({y},{x})"
                    ),
                )

                assert_grid_equal(
                    ref_v,
                    opt_v,
                    (
                        "full decoder v, "
                        f"d={d}, H({y},{x})"
                    ),
                )

        # Every vertical weight-one error.
        for y in range(d):
            for x in range(d):

                h = empty_grid(d)
                v = empty_grid(d)

                v[y][x] = True

                ref_h, ref_v = (
                    ref.decode_code_capacity(
                        h,
                        v,
                    )
                )

                opt_h, opt_v = (
                    optimized_decode(
                        h,
                        v,
                    )
                )

                assert_grid_equal(
                    ref_h,
                    opt_h,
                    (
                        "full decoder h, "
                        f"d={d}, V({y},{x})"
                    ),
                )

                assert_grid_equal(
                    ref_v,
                    opt_v,
                    (
                        "full decoder v, "
                        f"d={d}, V({y},{x})"
                    ),
                )


# =============================================================================
# Complete decoder: random configurations
# =============================================================================

def test_full_decoder_random():
    """
    Compare exact final residual configurations after the complete d^2
    paper schedule.

    This is stronger than merely comparing logical outcomes.
    """

    rng = random.Random(987654321)

    cases = {
        3: 500,
        5: 300,
        7: 100,
    }

    probabilities = (
        0.02,
        0.05,
        0.10,
        0.20,
        0.50,
    )

    for d, ncases in cases.items():

        for trial in range(ncases):

            p = probabilities[
                trial % len(probabilities)
            ]

            h = random_grid(
                d,
                p,
                rng,
            )

            v = random_grid(
                d,
                p,
                rng,
            )

            ref_h, ref_v = (
                ref.decode_code_capacity(
                    h,
                    v,
                )
            )

            opt_h, opt_v = (
                optimized_decode(
                    h,
                    v,
                )
            )

            if (
                not grids_equal(
                    ref_h,
                    opt_h,
                )
                or not grids_equal(
                    ref_v,
                    opt_v,
                )
            ):
                raise AssertionError(
                    "\n"
                    "Full decoder equivalence failed\n"
                    f"d      = {d}\n"
                    f"trial  = {trial}\n"
                    f"p      = {p}\n"
                )


# =============================================================================
# Logical-sector helper
# =============================================================================

def reference_logical_error(h, v):
    d = len(h)

    lx = False
    ly = False

    for x in range(d):
        lx ^= v[0][x]

    for y in range(d):
        ly ^= h[y][0]

    return lx or ly


def test_logical_helper():
    rng = random.Random(24680)

    for d in (3, 5, 7, 9):

        for _ in range(200):

            h = random_grid(
                d,
                0.4,
                rng,
            )

            v = random_grid(
                d,
                0.4,
                rng,
            )

            a = reference_logical_error(
                h,
                v,
            )

            b = opt.logical_error(
                np.asarray(
                    h,
                    dtype=np.uint8,
                ),
                np.asarray(
                    v,
                    dtype=np.uint8,
                ),
            )

            assert bool(a) == bool(b)


# =============================================================================
# Monte Carlo residual interface smoke test
# =============================================================================

def test_residual_interface():
    """
    Basic structural check of simulate_code_capacity_residuals().
    """

    d = 5
    shots = 137

    groups = (
        opt.simulate_code_capacity_residuals(
            d=d,
            p=0.05,
            shots=shots,
            seed=123,
        )
    )

    assert sum(
        group["group_size"]
        for group in groups
    ) == shots

    # 137 = 64 + 64 + 9
    assert len(groups) == 3

    assert groups[0]["group_size"] == 64
    assert groups[1]["group_size"] == 64
    assert groups[2]["group_size"] == 9

    for group in groups:

        assert group["h"].shape == (
            d,
            d,
        )

        assert group["v"].shape == (
            d,
            d,
        )

        assert group["h"].dtype == np.uint64
        assert group["v"].dtype == np.uint64

        # No mask may contain trajectories outside valid_mask.
        invalid = (
            ~group["valid_mask"]
        )

        assert (
            group["syndrome_mask"]
            & invalid
        ) == 0

        assert (
            group["logical_mask"]
            & invalid
        ) == 0


# =============================================================================
# Runner
# =============================================================================

def run_tests():

    tests = [
        test_reset_schedule_equivalence,
        test_syndrome_equivalence,

        test_single_step_simple_cases,
        test_random_one_step_equivalence,
        test_multispin_lane_independence,

        test_full_decoder_weight_one,
        test_full_decoder_random,

        test_logical_helper,
        test_residual_interface,
    ]

    print()
    print(
        "SCALA2D reference/optimized equivalence tests"
    )
    print("=" * 64)

    for test in tests:

        test()

        print(
            f"PASS  {test.__name__}"
        )

    print("=" * 64)
    print(
        f"All {len(tests)} equivalence tests passed."
    )
    print()


if __name__ == "__main__":
    run_tests()
