"""
SCALA1D code-capacity finite-size scaling.

Scaling ansatz:

    p_L(p,d) = F[(p-p_c)d^(1/nu)]

with F represented by a cubic polynomial near the critical point.

Run:
    python -m scripts.plot_code_capacity_fss

Input:
    data/code_capacity/code_capacity_fss.csv

Outputs:
    figs/scala1d_cc_fss.pdf
    figs/scala1d_cc_fss.png
    figs/scala1d_cc_fss_fit_stability.csv
    figs/scala1d_cc_fss_chi2_stability.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


# =============================================================================
# Paths
# =============================================================================

DATA_FILE = Path(
    "data/code_capacity/code_capacity_fss.csv"
)

FIG_DIR = Path("figs")

FIG_PDF = (
    FIG_DIR
    / "scala1d_cc_fss.pdf"
)

FIG_PNG = (
    FIG_DIR
    / "scala1d_cc_fss.png"
)

STABILITY_CSV = (
    FIG_DIR
    / "scala1d_cc_fss_fit_stability.csv"
)

STABILITY_PNG = (
    FIG_DIR
    / "scala1d_cc_fss_chi2_stability.png"
)


# =============================================================================
# Plot style
# =============================================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 14,
    "legend.fontsize": 10,
})


# =============================================================================
# Scaling model
# =============================================================================

POLY_ORDER = 3


# =============================================================================
# Stability scan
# =============================================================================
#
# The data support a broad range of sensible fits. These scans document
# robustness against both the minimum included distance and the p-window.
# =============================================================================

FIT_SPECS = [
    # All available distances
    (21, 0.460, 0.540),
    (21, 0.470, 0.530),
    (21, 0.480, 0.520),
    (21, 0.485, 0.515),

    # Remove smallest distance
    (31, 0.460, 0.540),
    (31, 0.470, 0.530),
    (31, 0.480, 0.520),
    (31, 0.485, 0.515),

    # More conservative asymptotic selections
    (51, 0.460, 0.540),
    (51, 0.470, 0.530),
    (51, 0.480, 0.520),
    (51, 0.485, 0.515),

    (71, 0.460, 0.540),
    (71, 0.470, 0.530),
    (71, 0.480, 0.520),
    (71, 0.485, 0.515),
]


# =============================================================================
# Final selected collapse
# =============================================================================
#
# This gives a clean fit while retaining nine code distances:
#
#   d = 51, 71, 91, 111, 131, 151, 171, 201, 251
#
# On the supplied dataset it gives approximately:
#
#   p_c      = 0.50006
#   nu       = 2.006
#   chi2_red = 0.73
#
# =============================================================================

FINAL_D_MIN = 51
FINAL_P_MIN = 0.470
FINAL_P_MAX = 0.530


# =============================================================================
# Scaling functions
# =============================================================================

def scaling_x(
    p,
    d,
    pc,
    nu,
):
    return (
        (p - pc)
        * d ** (1.0 / nu)
    )


def polynomial(
    x,
    coeffs,
):

    y = np.zeros_like(
        x,
        dtype=float,
    )

    for k, a in enumerate(
        coeffs
    ):
        y += (
            a
            * x**k
        )

    return y


def model(
    params,
    p,
    d,
):

    pc = params[0]
    nu = params[1]
    coeffs = params[2:]

    x = scaling_x(
        p,
        d,
        pc,
        nu,
    )

    return polynomial(
        x,
        coeffs,
    )


def residuals(
    params,
    p,
    d,
    pL,
    se,
):

    return (
        model(
            params,
            p,
            d,
        )
        - pL
    ) / se


# =============================================================================
# Initial parameters and bounds
# =============================================================================

def initial_parameters():

    initial = [
        0.5,
        2.0,
    ]

    coeff_initial = [
        0.5,
        1.0,
        0.0,
        0.0,
    ]

    initial += (
        coeff_initial[
            : POLY_ORDER + 1
        ]
    )

    return np.asarray(
        initial,
        dtype=float,
    )


def parameter_bounds():

    lower = [
        0.45,
        0.5,
    ]

    lower += (
        [-np.inf]
        * (POLY_ORDER + 1)
    )

    upper = [
        0.55,
        6.0,
    ]

    upper += (
        [np.inf]
        * (POLY_ORDER + 1)
    )

    return (
        np.asarray(
            lower,
            dtype=float,
        ),
        np.asarray(
            upper,
            dtype=float,
        ),
    )


# =============================================================================
# Data loading
# =============================================================================

def load_data():

    df = pd.read_csv(
        DATA_FILE
    )

    required = {
        "d",
        "p",
        "pL",
        "se",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"Missing columns: "
            f"{sorted(missing)}"
        )

    df = df[
        np.isfinite(df["p"])
        & np.isfinite(df["pL"])
        & np.isfinite(df["se"])
        & (df["se"] > 0)
    ].copy()

    if df.empty:
        raise RuntimeError(
            "No usable data."
        )

    return df


# =============================================================================
# Single FSS fit
# =============================================================================

def fit_fss(
    df,
    d_min,
    p_min,
    p_max,
):

    data = df[
        (df["d"] >= d_min)
        & (df["p"] >= p_min)
        & (df["p"] <= p_max)
    ].copy()

    if data.empty:
        return None

    p = data[
        "p"
    ].to_numpy(
        dtype=float
    )

    d = data[
        "d"
    ].to_numpy(
        dtype=float
    )

    pL = data[
        "pL"
    ].to_numpy(
        dtype=float
    )

    se = data[
        "se"
    ].to_numpy(
        dtype=float
    )

    n_params = (
        2
        + POLY_ORDER
        + 1
    )

    if len(data) <= n_params:
        return None

    initial = initial_parameters()

    lower, upper = parameter_bounds()

    result = least_squares(
        residuals,
        initial,
        args=(
            p,
            d,
            pL,
            se,
        ),
        bounds=(
            lower,
            upper,
        ),
        max_nfev=20000,
    )

    if not result.success:
        return None

    pc = float(
        result.x[0]
    )

    nu = float(
        result.x[1]
    )

    coeffs = (
        result.x[2:].copy()
    )

    chi2 = float(
        np.sum(
            result.fun**2
        )
    )

    dof = (
        len(data)
        - len(result.x)
    )

    chi2_red = (
        chi2 / dof
        if dof > 0
        else np.nan
    )


    # =========================================================================
    # Parameter covariance
    # =========================================================================
    #
    # The supplied Monte Carlo standard errors are independently estimated
    # statistical uncertainties. Therefore we use
    #
    #     Cov = (J^T J)^(-1)
    #
    # and DO NOT multiply by chi2_red.
    # =========================================================================

    jtj = (
        result.jac.T
        @ result.jac
    )

    cov = np.linalg.pinv(
        jtj
    )

    errors = np.sqrt(
        np.maximum(
            np.diag(cov),
            0.0,
        )
    )

    pc_err = float(
        errors[0]
    )

    nu_err = float(
        errors[1]
    )

    return {
        "d_min": int(d_min),
        "p_min": float(p_min),
        "p_max": float(p_max),

        "n_points": int(
            len(data)
        ),

        "n_distances": int(
            data["d"].nunique()
        ),

        "pc": pc,
        "pc_err": pc_err,

        "nu": nu,
        "nu_err": nu_err,

        "chi2": chi2,
        "dof": int(dof),
        "chi2_red": chi2_red,

        "coeffs": coeffs,
        "fit": result,
        "data": data,
    }


# =============================================================================
# Stability scan
# =============================================================================

def run_stability_scan(
    df,
):

    results = []

    for (
        d_min,
        p_min,
        p_max,
    ) in FIT_SPECS:

        result = fit_fss(
            df,
            d_min=d_min,
            p_min=p_min,
            p_max=p_max,
        )

        if result is None:
            continue

        results.append(
            result
        )

    if not results:
        raise RuntimeError(
            "No FSS fits succeeded."
        )

    return results


# =============================================================================
# Print stability scan
# =============================================================================

def print_stability_table(
    results,
):

    print()
    print("=" * 110)

    print(
        "SCALA1D CODE-CAPACITY "
        "FSS STABILITY SCAN"
    )

    print("=" * 110)

    header = (
        f"{'dmin':>6s} "
        f"{'pmin':>8s} "
        f"{'pmax':>8s} "
        f"{'N':>5s} "
        f"{'Nd':>4s} "
        f"{'pc':>11s} "
        f"{'pc err':>11s} "
        f"{'nu':>9s} "
        f"{'nu err':>9s} "
        f"{'chi2/dof':>10s}"
    )

    print(header)
    print("-" * len(header))

    for r in results:

        print(
            f"{r['d_min']:6d} "
            f"{r['p_min']:8.4f} "
            f"{r['p_max']:8.4f} "
            f"{r['n_points']:5d} "
            f"{r['n_distances']:4d} "
            f"{r['pc']:11.7f} "
            f"{r['pc_err']:11.7f} "
            f"{r['nu']:9.4f} "
            f"{r['nu_err']:9.4f} "
            f"{r['chi2_red']:10.3f}"
        )

    print("=" * 110)


# =============================================================================
# Save stability table
# =============================================================================

def save_stability_table(
    results,
):

    rows = []

    for r in results:

        rows.append({
            "d_min": r["d_min"],
            "p_min": r["p_min"],
            "p_max": r["p_max"],
            "n_points": r["n_points"],
            "n_distances": r["n_distances"],
            "pc": r["pc"],
            "pc_err": r["pc_err"],
            "nu": r["nu"],
            "nu_err": r["nu_err"],
            "chi2": r["chi2"],
            "dof": r["dof"],
            "chi2_red": r["chi2_red"],
        })

    pd.DataFrame(
        rows
    ).to_csv(
        STABILITY_CSV,
        index=False,
    )


# =============================================================================
# Plot chi2 stability
# =============================================================================

def plot_chi2_stability(
    results,
):

    labels = []
    chi2_values = []

    for r in results:

        labels.append(
            (
                rf"$d\geq{r['d_min']}$"
                "\n"
                rf"${r['p_min']:.3f}"
                "-"
                rf"{r['p_max']:.3f}$"
            )
        )

        chi2_values.append(
            r["chi2_red"]
        )

    x = np.arange(
        len(results)
    )

    fig, ax = plt.subplots(
        figsize=(6.5, 3.4)
    )

    ax.plot(
        x,
        chi2_values,
        linestyle="none",
        marker="o",
    )

    ax.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_ylabel(
        r"$\chi^2_{\rm red}$",
        fontsize=11,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
        fontsize=7.5,
    )

    ax.tick_params(
        axis="y",
        labelsize=9,
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        STABILITY_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# Get final selected fit
# =============================================================================

def get_final_fit(
    results,
    df,
):

    for r in results:

        if (
            r["d_min"] == FINAL_D_MIN
            and np.isclose(
                r["p_min"],
                FINAL_P_MIN,
            )
            and np.isclose(
                r["p_max"],
                FINAL_P_MAX,
            )
        ):
            return r

    return fit_fss(
        df,
        FINAL_D_MIN,
        FINAL_P_MIN,
        FINAL_P_MAX,
    )


# =============================================================================
# Final collapse
# =============================================================================

def plot_final_collapse(
    result,
):

    data = result["data"]

    pc = result["pc"]
    pc_err = result["pc_err"]

    nu = result["nu"]
    nu_err = result["nu_err"]

    chi2_red = (
        result["chi2_red"]
    )

    coeffs = result["coeffs"]

    distances = sorted(
        data["d"].unique()
    )

    fig = plt.figure(
        figsize=(5, 4)
    )

    ax = plt.gca()

    all_x = []
    all_y = []


    # =========================================================================
    # Numerical points
    # =========================================================================

    for distance in distances:

        subset = data[
            data["d"] == distance
        ].sort_values(
            "p"
        )

        p_sub = subset[
            "p"
        ].to_numpy(
            dtype=float
        )

        y_sub = subset[
            "pL"
        ].to_numpy(
            dtype=float
        )

        se_sub = subset[
            "se"
        ].to_numpy(
            dtype=float
        )

        x_sub = scaling_x(
            p_sub,
            float(distance),
            pc,
            nu,
        )

        all_x.extend(
            x_sub
        )

        all_y.extend(
            y_sub
        )

        # Points + error bars only.
        ax.errorbar(
            x_sub,
            y_sub,
            yerr=se_sub,
            linestyle="none",
            marker="o",
            markerfacecolor="none",
            capsize=2,
            label=str(int(distance)),
            zorder=3,
        )


    # =========================================================================
    # Fitted scaling function
    # =========================================================================

    x_fit = np.linspace(
        min(all_x),
        max(all_x),
        600,
    )

    y_fit = polynomial(
        x_fit,
        coeffs,
    )

    ax.plot(
        x_fit,
        y_fit,
        "--",
        color="black",
        linewidth=1.5,
        zorder=2,
    )


    # =========================================================================
    # Labels
    # =========================================================================

    ax.set_xlabel(
        r"$(p-p_c)d^{1/\nu}$"
    )

    ax.set_ylabel(
        r"$p_L$"
    )

    ax.grid()


    # =========================================================================
    # Tight limits
    # =========================================================================

    x_min = min(all_x)
    x_max = max(all_x)

    y_min = min(all_y)
    y_max = max(all_y)

    x_range = (
        x_max - x_min
    )

    y_range = (
        y_max - y_min
    )

    ax.set_xlim(
        x_min - 0.04 * x_range,
        x_max + 0.04 * x_range,
    )

    ax.set_ylim(
        max(
            0.0,
            y_min - 0.08 * y_range,
        ),
        y_max + 0.08 * y_range,
    )


    # =========================================================================
    # Fit annotation
    # =========================================================================

    ax.text(
        0.04,
        0.96,
        (
            rf"$p_c={pc:.5f}\pm{pc_err:.5f}$"
            "\n"
            rf"$\nu={nu:.3f}\pm{nu_err:.3f}$"
            "\n"
            rf"$\chi^2_{{\rm red}}={chi2_red:.3f}$"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
    )


    # =========================================================================
    # Legend
    # =========================================================================

    ax.legend(
        title=r"$d$",
        loc="lower right",
        ncol=3,
    )


    # =========================================================================
    # Save
    # =========================================================================

    fig.savefig(
        FIG_PDF,
        bbox_inches="tight",
    )

    fig.savefig(
        FIG_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# Main
# =============================================================================

def main():

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

    results = run_stability_scan(
        df
    )

    print_stability_table(
        results
    )

    save_stability_table(
        results
    )

    plot_chi2_stability(
        results
    )

    final = get_final_fit(
        results,
        df,
    )

    if final is None:
        raise RuntimeError(
            "Selected final fit failed."
        )

    print()
    print("=" * 70)
    print(
        "SELECTED COLLAPSE FIT"
    )
    print("=" * 70)

    print(
        f"d >=        "
        f"{final['d_min']}"
    )

    print(
        f"p window    "
        f"[{final['p_min']}, "
        f"{final['p_max']}]"
    )

    print(
        f"N points    "
        f"{final['n_points']}"
    )

    print(
        f"N distances "
        f"{final['n_distances']}"
    )

    print()

    print(
        f"p_c         = "
        f"{final['pc']:.8f} "
        f"+/- "
        f"{final['pc_err']:.8f}"
    )

    print(
        f"nu          = "
        f"{final['nu']:.6f} "
        f"+/- "
        f"{final['nu_err']:.6f}"
    )

    print(
        f"chi2_red    = "
        f"{final['chi2_red']:.3f}"
    )

    print("=" * 70)

    plot_final_collapse(
        final
    )

    print()

    print(
        f"Saved: {FIG_PDF}"
    )

    print(
        f"Saved: {FIG_PNG}"
    )

    print(
        f"Saved: {STABILITY_CSV}"
    )

    print(
        f"Saved: {STABILITY_PNG}"
    )


if __name__ == "__main__":
    main()
