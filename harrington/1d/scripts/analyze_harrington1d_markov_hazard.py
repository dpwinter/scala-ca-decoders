# scripts/analyze_harrington1d_markov_hazard.py

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.check_harrington1d_hazard import (
    simulate,
    hazard_curve,
)


# =============================================================================
# Reduced Harrington1D block Markov model
# =============================================================================

def block_flip_probability(p: float) -> float:
    """
    Probability that majority voting on a 3-qubit level-0 block
    produces a block error:

        p_b = 3 p^2 (1-p) + p^3.
    """
    return 3.0 * p**2 * (1.0 - p) + p**3


def weight_transition_probability(
    n: int,
    k: int,
    l: int,
    p: float,
) -> float:
    """
    Probability to go from Hamming weight k to Hamming weight l
    when each of n independent bits flips with probability p.

    k = number of erroneous blocks before the step
    l = number of erroneous blocks after the step
    """

    total = 0.0

    # a = number of existing 1s flipped to 0
    # b = number of existing 0s flipped to 1
    #
    # l = k - a + b
    #
    # therefore:
    #
    # b = l - k + a

    for a in range(k + 1):

        b = l - k + a

        if b < 0 or b > n - k:
            continue

        n_flips = a + b

        ways = (
            math.comb(k, a)
            * math.comb(n - k, b)
        )

        total += (
            ways
            * p**n_flips
            * (1.0 - p)**(n - n_flips)
        )

    return total


def reduced_transient_matrix(
    d: int,
    p_phys: float,
):
    """
    Construct the transient matrix W of the reduced block-error
    Markov process used in Appendix A/C.

    There are

        n_b = d / 3

    level-1 blocks.

    Logical failure occurs when more than half the blocks
    are erroneous.
    """

    if d % 3 != 0:
        raise ValueError(
            "Harrington1D distance must be divisible by 3."
        )

    n_blocks = d // 3

    p_block = block_flip_probability(
        p_phys
    )

    first_absorbing_weight = (
        n_blocks // 2 + 1
    )

    transient_weights = np.arange(
        first_absorbing_weight,
        dtype=int,
    )

    n_transient = len(
        transient_weights
    )

    W = np.zeros(
        (
            n_transient,
            n_transient,
        ),
        dtype=float,
    )

    for i, k in enumerate(
        transient_weights
    ):

        for j, l in enumerate(
            transient_weights
        ):

            W[i, j] = (
                weight_transition_probability(
                    n_blocks,
                    int(k),
                    int(l),
                    p_block,
                )
            )

    return (
        W,
        p_block,
        n_blocks,
        transient_weights,
    )


# =============================================================================
# Spectral quantities
# =============================================================================

def markov_quantities(
    d: int,
    p_phys: float,
):
    (
        W,
        p_block,
        n_blocks,
        transient_weights,
    ) = reduced_transient_matrix(
        d,
        p_phys,
    )

    # -------------------------------------------------------------------------
    # Eigenvalues
    # -------------------------------------------------------------------------

    eigvals = np.linalg.eigvals(
        W
    )

    order = np.argsort(
        -np.abs(
            eigvals
        )
    )

    eigvals = eigvals[
        order
    ]

    lambda1 = float(
        np.real(
            eigvals[0]
        )
    )

    if len(eigvals) > 1:

        lambda2 = float(
            np.abs(
                eigvals[1]
            )
        )

    else:

        lambda2 = 0.0

    # -------------------------------------------------------------------------
    # Quasi-stationary hazard
    # -------------------------------------------------------------------------

    h_spectral = (
        1.0
        - lambda1
    )

    # -------------------------------------------------------------------------
    # Exact mean hitting time starting from weight zero
    #
    # <T> = pi0 (I-W)^(-1) 1
    # -------------------------------------------------------------------------

    pi0 = np.zeros(
        len(W),
        dtype=float,
    )

    pi0[0] = 1.0

    ones = np.ones(
        len(W),
        dtype=float,
    )

    mean_lifetime = float(
        pi0
        @ np.linalg.solve(
            np.eye(
                len(W)
            )
            - W,
            ones,
        )
    )

    inverse_mean = (
        1.0
        / mean_lifetime
    )

    # -------------------------------------------------------------------------
    # Spectral relaxation timescale
    # -------------------------------------------------------------------------

    if (
        lambda2 > 0.0
        and lambda2 < lambda1
    ):

        tau_mix = (
            1.0
            / -math.log(
                lambda2
                / lambda1
            )
        )

    else:

        tau_mix = np.nan

    return {
        "W": W,
        "p_block": p_block,
        "n_blocks": n_blocks,
        "weights": transient_weights,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "h_spectral": h_spectral,
        "mean_lifetime": mean_lifetime,
        "inverse_mean": inverse_mean,
        "tau_mix": tau_mix,
    }


# =============================================================================
# Exact time-dependent hazard of reduced Markov model
# =============================================================================

def markov_hazard_curve(
    W,
    max_time: int,
):
    """
    Exact reduced-model survival and hazard:

        S(t) = pi0 W^t 1

        h(t) = 1 - S(t+1) / S(t).
    """

    n = W.shape[0]

    pi = np.zeros(
        n,
        dtype=float,
    )

    pi[0] = 1.0

    ones = np.ones(
        n,
        dtype=float,
    )

    times = []
    hazards = []
    survivals = []

    survival = float(
        pi @ ones
    )

    for t in range(
        max_time
    ):

        pi_next = (
            pi @ W
        )

        survival_next = float(
            pi_next
            @ ones
        )

        if survival <= 0.0:
            break

        hazard = (
            1.0
            - survival_next
            / survival
        )

        times.append(
            t
        )

        hazards.append(
            hazard
        )

        survivals.append(
            survival
        )

        pi = pi_next
        survival = survival_next

    return (
        np.asarray(
            times
        ),
        np.asarray(
            hazards
        ),
        np.asarray(
            survivals
        ),
    )


# =============================================================================
# Automatic plateau detection
# =============================================================================

def find_plateau_start(
    h,
    se,
    min_consecutive=4,
    z_max=2.0,
):
    """
    Find the earliest bin from which the hazard is statistically
    compatible with a constant plateau.

    For every candidate starting index i, fit the next
    min_consecutive bins to a constant using inverse-variance
    weighting. Accept the first candidate for which all bins lie
    within z_max standard deviations of that constant.
    """

    n = len(h)

    if n < min_consecutive:
        return None

    for i in range(
        n
        - min_consecutive
        + 1
    ):

        h_test = h[
            i:
            i + min_consecutive
        ]

        se_test = se[
            i:
            i + min_consecutive
        ]

        if (
            np.any(
                ~np.isfinite(
                    h_test
                )
            )
            or np.any(
                ~np.isfinite(
                    se_test
                )
            )
            or np.any(
                se_test <= 0
            )
        ):
            continue

        weights = (
            1.0
            / se_test**2
        )

        h_const = (
            np.sum(
                weights
                * h_test
            )
            / np.sum(
                weights
            )
        )

        z = (
            np.abs(
                h_test
                - h_const
            )
            / se_test
        )

        if np.all(
            z <= z_max
        ):

            return i

    return None


# =============================================================================
# Full-CA stationary hazard
# =============================================================================

def monte_carlo_hazard(
    d,
    p,
    shots,
    rounds,
    bin_width,
    min_at_risk,
    seed,
):
    """
    Run full Harrington1D CA under PURE DATA NOISE q=0 and
    automatically estimate its stationary hazard plateau.
    """

    failure_times = simulate(
        d=d,
        p=p,
        q=0.0,
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
        [
            row[0]
            for row in curve
        ]
    )

    h = np.asarray(
        [
            row[1]
            for row in curve
        ]
    )

    se = np.asarray(
        [
            row[2]
            for row in curve
        ]
    )

    plateau_index = find_plateau_start(
        h,
        se,
    )

    if plateau_index is None:
        return None

    # -------------------------------------------------------------------------
    # Fit all statistically resolved bins after plateau onset
    # -------------------------------------------------------------------------

    h_fit = h[
        plateau_index:
    ]

    se_fit = se[
        plateau_index:
    ]

    valid = (
        np.isfinite(
            h_fit
        )
        & np.isfinite(
            se_fit
        )
        & (
            se_fit > 0
        )
    )

    h_fit = h_fit[
        valid
    ]

    se_fit = se_fit[
        valid
    ]

    if len(
        h_fit
    ) == 0:
        return None

    weights = (
        1.0
        / se_fit**2
    )

    h_plateau = float(
        np.sum(
            weights
            * h_fit
        )
        / np.sum(
            weights
        )
    )

    h_plateau_se = float(
        1.0
        / np.sqrt(
            np.sum(
                weights
            )
        )
    )

    plateau_time = float(
        t[
            plateau_index
        ]
    )

    return {
        "t": t,
        "h": h,
        "se": se,
        "plateau_index": int(
            plateau_index
        ),
        "plateau_time": plateau_time,
        "h_plateau": h_plateau,
        "h_plateau_se": h_plateau_se,
        "failed": int(
            np.count_nonzero(
                failure_times
            )
        ),
    }


# =============================================================================
# Main
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--d",
        type=int,
        default=9,
    )

    parser.add_argument(
        "--p",
        nargs="+",
        type=float,
        default=[
            0.05,
            0.075,
            0.10,
            0.15,
            0.20,
        ],
    )

    parser.add_argument(
        "--shots",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=5_000,
    )

    parser.add_argument(
        "--bin-width",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--min-at-risk",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "figs/harrington1d_markov_hazard_test.pdf"
        ),
    )

    args = parser.parse_args()

    # =========================================================================
    # Numerical comparison
    # =========================================================================

    results = []

    print()

    print("=" * 136)

    print(
        "HARRINGTON1D: REDUCED MARKOV MODEL VS FULL-CA "
        "STATIONARY HAZARD -- PURE DATA NOISE"
    )

    print("=" * 136)

    print(
        f"{'d':>4s} "
        f"{'p':>8s} "
        f"{'p_b':>10s} "
        f"{'rho(W)':>12s} "
        f"{'1-rho':>12s} "
        f"{'1/<T>':>12s} "
        f"{'h_MC':>12s} "
        f"{'SE_MC':>12s} "
        f"{'MC/(1-rho)':>14s} "
        f"{'t_plateau':>12s} "
        f"{'tau_mix':>10s}"
    )

    print(
        "-" * 136
    )

    for j, p in enumerate(
        args.p
    ):

        analytic = markov_quantities(
            args.d,
            p,
        )

        mc = monte_carlo_hazard(
            d=args.d,
            p=p,
            shots=args.shots,
            rounds=args.rounds,
            bin_width=args.bin_width,
            min_at_risk=args.min_at_risk,
            seed=args.seed + j,
        )

        if mc is None:

            h_mc = np.nan
            h_mc_se = np.nan
            ratio = np.nan
            plateau_time = np.nan

        else:

            h_mc = (
                mc[
                    "h_plateau"
                ]
            )

            h_mc_se = (
                mc[
                    "h_plateau_se"
                ]
            )

            plateau_time = (
                mc[
                    "plateau_time"
                ]
            )

            ratio = (
                h_mc
                / analytic[
                    "h_spectral"
                ]
            )

        print(
            f"{args.d:4d} "
            f"{p:8.5f} "
            f"{analytic['p_block']:10.4e} "
            f"{analytic['lambda1']:12.8f} "
            f"{analytic['h_spectral']:12.4e} "
            f"{analytic['inverse_mean']:12.4e} "
            f"{h_mc:12.4e} "
            f"{h_mc_se:12.4e} "
            f"{ratio:14.5f} "
            f"{plateau_time:12.1f} "
            f"{analytic['tau_mix']:10.2f}"
        )

        results.append(
            (
                p,
                analytic,
                mc,
            )
        )

    # =========================================================================
    # Figure 1:
    # logical failure rates vs physical p
    # =========================================================================

    p_values = np.asarray(
        [
            r[0]
            for r in results
        ]
    )

    h_spectral = np.asarray(
        [
            r[1][
                "h_spectral"
            ]
            for r in results
        ]
    )

    inverse_mean = np.asarray(
        [
            r[1][
                "inverse_mean"
            ]
            for r in results
        ]
    )

    h_mc = np.asarray(
        [
            (
                r[2][
                    "h_plateau"
                ]
                if r[2] is not None
                else np.nan
            )
            for r in results
        ]
    )

    h_mc_se = np.asarray(
        [
            (
                r[2][
                    "h_plateau_se"
                ]
                if r[2] is not None
                else np.nan
            )
            for r in results
        ]
    )

    fig, ax = plt.subplots(
        figsize=(
            5,
            4,
        )
    )

    ax.plot(
        p_values,
        h_spectral,
        "o-",
        label=r"$1-\rho(W)$",
    )

    ax.plot(
        p_values,
        inverse_mean,
        "s--",
        label=r"$1/\langle T\rangle_{\rm Markov}$",
    )

    valid_mc = (
        np.isfinite(
            h_mc
        )
    )

    ax.errorbar(
        p_values[
            valid_mc
        ],
        h_mc[
            valid_mc
        ],
        yerr=h_mc_se[
            valid_mc
        ],
        fmt="o",
        markerfacecolor="none",
        capsize=3,
        label=r"$h_\infty^{\rm full\ CA}$",
    )

    ax.set_xscale(
        "log"
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        r"data error rate $p$"
    )

    ax.set_ylabel(
        r"logical failure rate"
    )

    ax.grid(
        True,
        which="major",
    )

    ax.grid(
        False,
        which="minor",
    )

    ax.legend()

    ax.text(
        0.05,
        0.95,
        rf"$d={args.d},\ q=0$",
        transform=ax.transAxes,
        va="top",
    )

    fig.tight_layout()

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        args.output,
        bbox_inches="tight",
    )

    fig.savefig(
        args.output.with_suffix(
            ".png"
        ),
        dpi=300,
        bbox_inches="tight",
    )

    # =========================================================================
    # Figure 2:
    # reduced-Markov hazard relaxation
    # =========================================================================

    fig2, ax2 = plt.subplots(
        figsize=(
            5,
            4,
        )
    )

    for (
        p,
        analytic,
        _,
    ) in results:

        (
            t,
            h,
            _,
        ) = markov_hazard_curve(
            analytic[
                "W"
            ],
            max_time=min(
                args.rounds,
                5000,
            ),
        )

        ax2.plot(
            t,
            h,
            label=rf"$p={p:g}$",
        )

        ax2.axhline(
            analytic[
                "h_spectral"
            ],
            linestyle="--",
            linewidth=1,
        )

    ax2.set_yscale(
        "log"
    )

    ax2.set_xlabel(
        r"Markov step $t$"
    )

    ax2.set_ylabel(
        r"$h_{\rm Markov}(t)$"
    )

    ax2.grid(
        True,
        which="major",
    )

    ax2.grid(
        False,
        which="minor",
    )

    ax2.legend(
        title=rf"$d={args.d}$"
    )

    fig2.tight_layout()

    second_output = (
        args.output.parent
        / (
            args.output.stem
            + "_relaxation"
            + args.output.suffix
        )
    )

    fig2.savefig(
        second_output,
        bbox_inches="tight",
    )

    fig2.savefig(
        second_output.with_suffix(
            ".png"
        ),
        dpi=300,
        bbox_inches="tight",
    )

    # =========================================================================
    # Figure 3:
    # full-CA hazard curves with automatically detected plateau onset
    # =========================================================================

    fig3, ax3 = plt.subplots(
        figsize=(
            5,
            4,
        )
    )

    for (
        p,
        _,
        mc,
    ) in results:

        if mc is None:
            continue

        ax3.errorbar(
            mc["t"],
            mc["h"],
            yerr=mc["se"],
            marker="o",
            markerfacecolor="none",
            linestyle="-",
            capsize=2,
            label=rf"$p={p:g}$",
        )

        ax3.axvline(
            mc[
                "plateau_time"
            ],
            linestyle=":",
            linewidth=1,
        )

        ax3.axhline(
            mc[
                "h_plateau"
            ],
            linestyle="--",
            linewidth=1,
        )

    ax3.set_yscale(
        "log"
    )

    ax3.set_xlabel(
        r"time $t$"
    )

    ax3.set_ylabel(
        r"full-CA hazard $h(t)$"
    )

    ax3.grid(
        True,
        which="major",
    )

    ax3.grid(
        False,
        which="minor",
    )

    ax3.legend(
        title=rf"$d={args.d},\ q=0$"
    )

    fig3.tight_layout()

    third_output = (
        args.output.parent
        / (
            args.output.stem
            + "_full_ca_plateaus"
            + args.output.suffix
        )
    )

    fig3.savefig(
        third_output,
        bbox_inches="tight",
    )

    fig3.savefig(
        third_output.with_suffix(
            ".png"
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()
