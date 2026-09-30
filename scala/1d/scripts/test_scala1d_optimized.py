"""
Verify that src/scala1d.py and src/scala1d_optimized.py
implement exactly the same SCALA1D update.

Run from the scala1d project root:

    python -m scripts.test_scala1d_equivalence
"""

from itertools import product

import numpy as np

from src.scala1d import step as reference_step
from src.scala1d import syndrome

from src.scala1d_optimized import scala_step


# ============================================================
# Packing helpers
# ============================================================

def pack(states):
    """
    Pack Boolean trajectories into uint64 bit slices.

    states has shape

        (number_of_lanes, d)

    and the returned array has shape

        (d,).
    """

    lanes, d = states.shape

    packed = np.zeros(d, dtype=np.uint64)

    for lane in range(lanes):
        bit = np.uint64(1) << np.uint64(lane)

        for i in range(d):
            if states[lane, i]:
                packed[i] |= bit

    return packed


def unpack(packed, lanes):
    """Inverse of pack()."""

    d = len(packed)

    states = np.zeros(
        (lanes, d),
        dtype=bool,
    )

    for lane in range(lanes):
        bit = np.uint64(1) << np.uint64(lane)

        for i in range(d):
            states[lane, i] = bool(
                packed[i] & bit
            )

    return states


# ============================================================
# Compare one packed batch
# ============================================================

def compare_batch(errors, defect, left, right):

    lanes, d = errors.shape

    # -------------------------------
    # Reference implementation
    # -------------------------------

    ref_errors = np.zeros_like(errors)
    ref_left = np.zeros_like(left)
    ref_right = np.zeros_like(right)

    for lane in range(lanes):

        e, l, r = reference_step(
            errors[lane].tolist(),
            defect[lane].tolist(),
            left[lane].tolist(),
            right[lane].tolist(),
        )

        ref_errors[lane] = e
        ref_left[lane] = l
        ref_right[lane] = r

    # -------------------------------
    # Optimized packed implementation
    # -------------------------------

    e_packed = pack(errors)
    d_packed = pack(defect)
    l_packed = pack(left)
    r_packed = pack(right)

    new_left = np.empty(
        d,
        dtype=np.uint64,
    )

    new_right = np.empty(
        d,
        dtype=np.uint64,
    )

    scala_step(
        e_packed,
        l_packed,
        r_packed,
        d_packed,
        new_left,
        new_right,
        d,
    )

    opt_errors = unpack(
        e_packed,
        lanes,
    )

    opt_left = unpack(
        l_packed,
        lanes,
    )

    opt_right = unpack(
        r_packed,
        lanes,
    )

    # -------------------------------
    # Exact comparison
    # -------------------------------

    if not np.array_equal(
        ref_errors,
        opt_errors,
    ):
        raise AssertionError(
            "Error configurations differ."
        )

    if not np.array_equal(
        ref_left,
        opt_left,
    ):
        raise AssertionError(
            "Left signals differ."
        )

    if not np.array_equal(
        ref_right,
        opt_right,
    ):
        raise AssertionError(
            "Right signals differ."
        )


# ============================================================
# Exhaustive d=3 test
# ============================================================

def test_exhaustive_d3():
    """
    Exhaustively test every possible

        errors, syndrome, left, right

    configuration at d=3.

    Number of states:

        2^(4d) = 4096.
    """

    d = 3

    states = list(
        product(
            [False, True],
            repeat=4 * d,
        )
    )

    batch_size = 64

    for start in range(
        0,
        len(states),
        batch_size,
    ):

        batch = states[
            start:start + batch_size
        ]

        lanes = len(batch)

        errors = np.zeros(
            (lanes, d),
            dtype=bool,
        )

        defect = np.zeros(
            (lanes, d),
            dtype=bool,
        )

        left = np.zeros(
            (lanes, d),
            dtype=bool,
        )

        right = np.zeros(
            (lanes, d),
            dtype=bool,
        )

        for lane, state in enumerate(batch):

            errors[lane] = state[0:d]
            defect[lane] = state[d:2*d]
            left[lane] = state[2*d:3*d]
            right[lane] = state[3*d:4*d]

        compare_batch(
            errors,
            defect,
            left,
            right,
        )

    print(
        "d=3 exhaustive transition test: "
        "4096 / 4096 passed"
    )


# ============================================================
# Random larger-distance tests
# ============================================================

def test_random_transitions():
    """
    Random arbitrary CA states at larger d.
    """

    rng = np.random.default_rng(12345)

    for d in [5, 7, 9, 15, 31]:

        for _ in range(100):

            lanes = 64

            errors = rng.integers(
                0,
                2,
                size=(lanes, d),
                dtype=np.uint8,
            ).astype(bool)

            defect = rng.integers(
                0,
                2,
                size=(lanes, d),
                dtype=np.uint8,
            ).astype(bool)

            left = rng.integers(
                0,
                2,
                size=(lanes, d),
                dtype=np.uint8,
            ).astype(bool)

            right = rng.integers(
                0,
                2,
                size=(lanes, d),
                dtype=np.uint8,
            ).astype(bool)

            compare_batch(
                errors,
                defect,
                left,
                right,
            )

        print(
            f"d={d}: "
            "6400 random transitions passed"
        )


# ============================================================
# Multi-step physical trajectories
# ============================================================

def test_random_trajectories():
    """
    Compare complete noise-free trajectories.

    Here the syndrome is recomputed from the evolving
    physical error configuration after every step.
    """

    rng = np.random.default_rng(67890)

    for d in [5, 7, 9, 15]:

        for trial in range(100):

            initial = rng.integers(
                0,
                2,
                size=d,
                dtype=np.uint8,
            ).astype(bool)

            # -----------------------
            # Reference
            # -----------------------

            e_ref = initial.tolist()
            l_ref = [False] * d
            r_ref = [False] * d

            # -----------------------
            # Optimized: one lane
            # -----------------------

            e_opt = pack(
                initial.reshape(1, d)
            )

            l_opt = np.zeros(
                d,
                dtype=np.uint64,
            )

            r_opt = np.zeros(
                d,
                dtype=np.uint64,
            )

            s_opt = np.empty(
                d,
                dtype=np.uint64,
            )

            ln_opt = np.empty(
                d,
                dtype=np.uint64,
            )

            rn_opt = np.empty(
                d,
                dtype=np.uint64,
            )

            for t in range(d):

                # Reference syndrome.
                s_ref = syndrome(e_ref)

                e_ref, l_ref, r_ref = reference_step(
                    e_ref,
                    s_ref,
                    l_ref,
                    r_ref,
                )

                # Packed syndrome.
                for i in range(d):

                    im = (i - 1) % d

                    s_opt[i] = (
                        e_opt[im]
                        ^ e_opt[i]
                    )

                scala_step(
                    e_opt,
                    l_opt,
                    r_opt,
                    s_opt,
                    ln_opt,
                    rn_opt,
                    d,
                )

                e_check = unpack(
                    e_opt,
                    1,
                )[0]

                l_check = unpack(
                    l_opt,
                    1,
                )[0]

                r_check = unpack(
                    r_opt,
                    1,
                )[0]

                assert np.array_equal(
                    e_check,
                    np.asarray(e_ref),
                ), (
                    f"errors differ at "
                    f"d={d}, trial={trial}, t={t}"
                )

                assert np.array_equal(
                    l_check,
                    np.asarray(l_ref),
                ), (
                    f"left signals differ at "
                    f"d={d}, trial={trial}, t={t}"
                )

                assert np.array_equal(
                    r_check,
                    np.asarray(r_ref),
                ), (
                    f"right signals differ at "
                    f"d={d}, trial={trial}, t={t}"
                )

        print(
            f"d={d}: "
            "100 full trajectories passed"
        )


# ============================================================
# Run tests
# ============================================================

if __name__ == "__main__":

    test_exhaustive_d3()
    test_random_transitions()
    test_random_trajectories()

    print()
    print(
        "All reference/optimized "
        "SCALA1D equivalence tests passed."
    )
