from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scripts.check_harrington1d_hazard import (
    simulate,
    hazard_curve,
)


# =============================================================================
# Plateau finder
# =============================================================================

def find_plateau_start(
    h,
    se,
    min_consecutive=4,
    z_max=2.0,
):
    n = len(h)

    if n < min_consecutive:
        return None

    for i in range(
        n - min_consecutive + 1
    ):

        hh = h[
            i:i + min_consecutive
        ]

        ss = se[
            i:i + min_consecutive
        ]

        if (
            np.any(~np.isfinite(hh))
            or np.any(~np.isfinite(ss))
            or np.any(ss <= 0)
        ):
            continue

        w = 1.0 / ss**2

        mean = (
            np.sum(w * hh)
            / np.sum(w)
        )

        z = (
            np.abs(hh - mean)
            / ss
        )

        if np.all(
            z <= z_max
        ):
            return i

    return None


def full_ca_stationary_hazard(
    d,
    p,
    q,
    shots,
    rounds,
    bin_width,
    min_at_risk,
    seed,
):
    failure_times = simulate(
        d=d,
        p=p,
        q=q,
        shots=shots,
        rounds=rounds,
        seed=seed,
    )

    curve = hazard_curve(
        failure_times,
        rounds,
        bin_width,
        min_at_risk,
    )

    if not curve:
        return None

    t = np.asarray(
        [row[0] for row in curve]
    )

    h = np.asarray(
        [row[1] for row in curve]
    )

    se = np.asarray(
        [row[2] for row in curve]
    )

    i0 = find_plateau_start(
        h,
        se,
    )

    if i0 is None:
        return None

    h_fit = h[i0:]
    se_fit = se[i0:]

    valid = (
        np.isfinite(h_fit)
        & np.isfinite(se_fit)
        & (se_fit > 0)
    )

    h_fit = h_fit[valid]
    se_fit = se_fit[valid]

    if len(h_fit) == 0:
        return None

    w = 1.0 / se_fit**2

    h_plateau = float(
        np.sum(
            w * h_fit
        )
        / np.sum(w)
    )

    h_plateau_se = float(
        1.0
        / np.sqrt(
            np.sum(w)
        )
    )

    return {
        "h": h_plateau,
        "se": h_plateau_se,
        "t_plateau": float(
            t[i0]
        ),
        "failed": int(
            np.count_nonzero(
                failure_times
            )
        ),
    }


# =============================================================================
# d=9 cycle matrix
# =============================================================================

def load_d9_hazard(path):
    data = np.load(
        path
    )

    K = np.asarray(
        data["K"],
        dtype=float,
    )

    tau1 = int(
        data["tau"]
    )

    p = float(
        data["p"]
    )

    q = float(
        data["q"]
    )

    eigvals = np.linalg.eigvals(
        K
    )

    rho = float(
        np.max(
            np.abs(eigvals)
        )
    )

    h9 = (
        1.0
        - rho**(
            1.0 / tau1
        )
    )

    return {
        "K": K,
        "tau1": tau1,
        "rho": rho,
        "h9": h9,
        "p": p,
        "q": q,
    }


# =============================================================================
# Hierarchical recursion
# =============================================================================

def majority_failure_probability(r):
    """
    Probability that at least two of three effective blocks fail.
    """
    return (
        3.0
        * r**2
        * (1.0 - r)
        + r**3
    )


def predict_d27_from_d9(
    h9,
    U=10,
):
    """
    First scalar hierarchical recursion.

    Level-2 period:
        tau2 = U^2 + 3^2.

    Treat the d=9 stationary hazard as a geometric effective
    block-failure process over that time interval.
    """

    tau2 = (
        U**2
        + 3**2
    )

    # Probability that one effective d=9 block fails during
    # one level-2 cycle.
    r9 = (
        1.0
        - (1.0 - h9)**tau2
    )

    # Failure of at least two of three effective blocks.
    p_cycle_27 = (
        majority_failure_probability(
            r9
        )
    )

    # Convert cycle survival back to elementary-step hazard.
    h27 = (
        1.0
        - (
            1.0
            - p_cycle_27
        )**(
            1.0 / tau2
        )
    )

    return {
        "tau2": tau2,
        "r9": r9,
        "p_cycle_27": p_cycle_27,
        "h27": h27,
    }


# =============================================================================
# Main
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--d9-matrix",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--shots",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=20_000,
    )

    parser.add_argument(
        "--bin-width",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--min-at-risk",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    # =========================================================================
    # d=9 reduced Markov result
    # =========================================================================

    d9 = load_d9_hazard(
        args.d9_matrix
    )

    prediction = (
        predict_d27_from_d9(
            d9["h9"]
        )
    )

    p = d9["p"]
    q = d9["q"]

    # =========================================================================
    # Full d=27 simulation
    # =========================================================================

    print()
    print("=" * 88)
    print(
        "HARRINGTON1D HIERARCHICAL RECURSION TEST"
    )
    print("=" * 88)

    print(
        f"p = {p:.6f}"
    )

    print(
        f"q = {q:.6f}"
    )

    print()

    print(
        "d=9 reduced cycle model"
    )

    print(
        f"    tau1                = "
        f"{d9['tau1']}"
    )

    print(
        f"    rho(K9)             = "
        f"{d9['rho']:.10f}"
    )

    print(
        f"    h9 / CA step        = "
        f"{d9['h9']:.6e}"
    )

    print()

    print(
        "naive level-2 recursion"
    )

    print(
        f"    tau2                = "
        f"{prediction['tau2']}"
    )

    print(
        f"    block fail prob r9  = "
        f"{prediction['r9']:.6e}"
    )

    print(
        f"    P27 per level-2 cyc = "
        f"{prediction['p_cycle_27']:.6e}"
    )

    print(
        f"    predicted h27       = "
        f"{prediction['h27']:.6e}"
    )

    print()
    print(
        "running full d=27 CA ..."
    )

    mc = full_ca_stationary_hazard(
        d=27,
        p=p,
        q=q,
        shots=args.shots,
        rounds=args.rounds,
        bin_width=args.bin_width,
        min_at_risk=args.min_at_risk,
        seed=args.seed,
    )

    print()

    if mc is None:

        print(
            "Could not resolve a stationary "
            "d=27 hazard."
        )

        return

    ratio = (
        mc["h"]
        / prediction[
            "h27"
        ]
    )

    print(
        "full d=27 CA"
    )

    print(
        f"    h27_MC              = "
        f"{mc['h']:.6e}"
    )

    print(
        f"    SE                   = "
        f"{mc['se']:.6e}"
    )

    print(
        f"    plateau begins       = "
        f"{mc['t_plateau']:.1f}"
    )

    print(
        f"    failures             = "
        f"{mc['failed']}"
    )

    print()

    print(
        "comparison"
    )

    print(
        f"    MC / prediction      = "
        f"{ratio:.4f}"
    )

    print(
        f"    relative difference  = "
        f"{100.0 * (ratio - 1.0):+.2f}%"
    )

    print()
    print("=" * 88)


if __name__ == "__main__":
    main()
