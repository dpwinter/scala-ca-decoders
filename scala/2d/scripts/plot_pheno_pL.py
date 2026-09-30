# scripts/plot_pheno_pL.py

from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.ticker import (
    LogLocator,
    LogFormatterMathtext,
    NullFormatter,
)
import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

INPUT = Path(
    "data/pheno/scala2d_pheno_pL_staircase.csv"
)

OUTPUT_DIR = Path("figs")

OUTPUT_PDF = (
    OUTPUT_DIR
    / "scala2d_pheno_pL_staircase.pdf"
)

OUTPUT_PNG = (
    OUTPUT_DIR
    / "scala2d_pheno_pL_staircase.png"
)

OUTPUT_FITS = (
    OUTPUT_DIR
    / "scala2d_pheno_pL_staircase_fits.csv"
)


# =============================================================================
# Distances shown in the main-paper figure
# =============================================================================
#
# Only include sizes for which the crossing progression is visible in the
# accessible p range.
# =============================================================================

PLOT_D = [
    5,
    9,
    15,
    21,
    31,
]


# =============================================================================
# Distances for which we report a low-p exponent fit
# =============================================================================
#
# d=31 is deliberately omitted from the exponent inset:
# its lowest resolved points still give a visibly pre-asymptotic exponent
# below the exact staircase prediction w0(31)=12.
# =============================================================================

FIT_D = [
    5,
    9,
    15,
    21,
]


# =============================================================================
# Number of lowest resolved points used for each free power-law fit
# =============================================================================
#
# These windows are deliberately short and restricted to the low-p tail.
# They are analogous to the conservative tail fits used for code capacity.
# =============================================================================

FIT_N_LOWEST = {
    5: 6,
    9: 5,
    15: 5,
    21: 3,
}


# =============================================================================
# Main-plot reduction
# =============================================================================
#
# Use all resolved data for fitting, but show only a reduced subset in the
# figure. This keeps the curves readable while preserving the numerics.
# =============================================================================

MAX_PLOT_POINTS = 11


# =============================================================================
# Axes
# =============================================================================

XMIN = 1.0e-3
XMAX = 2e-2

YMIN = 1.0e-8
YMAX = 1.0


# Inset only needs to cover the displayed distances.
INSET_D_MAX = 35


plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
})


# =============================================================================
# Exact discrete staircase prediction
# =============================================================================

def min_factor_sum(w):
    """
    f(w) = min{k+n : k,n positive integers and k*n = w}.
    """

    best = w + 1

    for k in range(1, math.isqrt(w) + 1):

        if w % k == 0:

            n = w // k

            best = min(
                best,
                k + n,
            )

    return best


def predicted_w0(d):
    """
    Exact discrete SCALA2D staircase:

        w0(d) = min { w in Z_+ :
                      2w - f(w) + 1 >= (d+1)/2 }.
    """

    target = (
        d + 1
    ) / 2.0

    w = 1

    while True:

        if (
            2 * w
            - min_factor_sum(w)
            + 1
            >= target
        ):
            return w

        w += 1


# =============================================================================
# Helpers
# =============================================================================

def as_bool(series):

    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
            ]
        )
    )


def weighted_linear_fit(
    X,
    y,
    sigma,
):

    weights = (
        1.0
        / sigma**2
    )

    XT_W = (
        X.T
        * weights
    )

    cov = np.linalg.pinv(
        XT_W @ X
    )

    beta = (
        cov
        @ XT_W
        @ y
    )

    residual = (
        y
        - X @ beta
    )

    chi2 = np.sum(
        (
            residual
            / sigma
        )**2
    )

    dof = (
        len(y)
        - X.shape[1]
    )

    chi2_red = (
        chi2 / dof
        if dof > 0
        else np.nan
    )

    return (
        beta,
        cov,
        chi2,
        chi2_red,
    )


def fit_lowest_points(
    sub,
    n_points,
):
    """
    Fit the n_points lowest-p resolved points to

        pL = A p^lambda

    using weighted least squares in log space.
    """

    sub = (
        sub[
            (sub["p"] > 0)
            & (sub["pL"] > 0)
            & (sub["pL_se"] > 0)
            & np.isfinite(
                sub["p"]
            )
            & np.isfinite(
                sub["pL"]
            )
            & np.isfinite(
                sub["pL_se"]
            )
        ]
        .sort_values("p")
        .copy()
    )

    if len(sub) < n_points:
        return None

    fit_sub = (
        sub.iloc[
            :n_points
        ]
        .copy()
    )

    p = (
        fit_sub["p"]
        .to_numpy(float)
    )

    pL = (
        fit_sub["pL"]
        .to_numpy(float)
    )

    se = (
        fit_sub["pL_se"]
        .to_numpy(float)
    )

    x = np.log(p)
    y = np.log(pL)

    sigma_log = (
        se / pL
    )

    X = np.column_stack(
        [
            np.ones_like(x),
            x,
        ]
    )

    beta, cov, chi2, chi2_red = (
        weighted_linear_fit(
            X,
            y,
            sigma_log,
        )
    )

    log_A, lam = beta

    return {
        "A": float(
            np.exp(log_A)
        ),
        "lambda": float(
            lam
        ),
        "lambda_err": float(
            np.sqrt(
                cov[1, 1]
            )
        ),
        "chi2": float(
            chi2
        ),
        "chi2_red": float(
            chi2_red
        ),
        "N": len(
            fit_sub
        ),
        "data": fit_sub,
        "p_min": float(
            p.min()
        ),
        "p_max": float(
            p.max()
        ),
    }


def power_law(
    p,
    A,
    lam,
):

    return (
        A
        * np.asarray(
            p,
            dtype=float,
        )**lam
    )


def reduced_plot_subset(
    sub,
    max_points=MAX_PLOT_POINTS,
):
    """
    Select a reduced set of actual Monte Carlo points for plotting.

    The selection is approximately uniform in log(p), always includes
    the first and last resolved points, and never interpolates data.
    """

    sub = (
        sub
        .sort_values("p")
        .reset_index(drop=True)
        .copy()
    )

    n = len(sub)

    if n <= max_points:
        return sub

    logp = np.log(
        sub["p"].to_numpy(float)
    )

    targets = np.linspace(
        logp.min(),
        logp.max(),
        max_points,
    )

    selected = []

    for target in targets:

        idx = int(
            np.argmin(
                np.abs(
                    logp - target
                )
            )
        )

        selected.append(
            idx
        )

    selected.extend(
        [
            0,
            n - 1,
        ]
    )

    selected = sorted(
        set(selected)
    )

    return (
        sub.iloc[
            selected
        ]
        .copy()
    )


# =============================================================================
# Load data
# =============================================================================

df = pd.read_csv(
    INPUT
)


required = {
    "d",
    "p",
    "q",
    "stationary",
    "pL",
    "pL_se",
}


missing = (
    required
    - set(
        df.columns
    )
)


if missing:

    raise RuntimeError(
        f"Missing columns: "
        f"{sorted(missing)}"
    )


# =============================================================================
# Keep only resolved baseline matched-noise points
# =============================================================================

stationary = as_bool(
    df["stationary"]
)


mask = (
    stationary
    & df["d"].isin(
        PLOT_D
    )
    & np.isclose(
        df["p"],
        df["q"],
        atol=1e-12,
        rtol=0.0,
    )
    & (
        df["p"] > 0.0
    )
    & (
        df["pL"] > 0.0
    )
    & (
        df["pL_se"] > 0.0
    )
    & np.isfinite(
        df["p"]
    )
    & np.isfinite(
        df["pL"]
    )
    & np.isfinite(
        df["pL_se"]
    )
)


if "p_sig" in df.columns:

    mask &= np.isclose(
        df["p_sig"],
        0.0,
        atol=1e-15,
        rtol=0.0,
    )


if "burn_reset" in df.columns:

    mask &= np.isclose(
        df["burn_reset"],
        20,
        atol=1e-12,
        rtol=0.0,
    )


df = (
    df[
        mask
    ]
    .copy()
)


# =============================================================================
# Diagnostics
# =============================================================================

print()
print("=" * 108)

print(
    "SCALA2D PHENOMENOLOGICAL MATCHED-NOISE "
    "LOW-p POWER-LAW FITS"
)

print("=" * 108)


for d in PLOT_D:

    sub = (
        df[
            df["d"] == d
        ]
        .sort_values("p")
        .copy()
    )

    if sub.empty:

        print(
            f"d={d:3d}: "
            "no resolved points"
        )

        continue

    print(
        f"d={d:3d}: "
        f"N_resolved={len(sub):2d}, "
        f"p=["
        f"{sub['p'].min():.6f}, "
        f"{sub['p'].max():.6f}"
        f"], "
        f"w0={predicted_w0(d)}"
    )


# =============================================================================
# Fits
# =============================================================================

fits = {}
fit_rows = []


print()
print("-" * 108)


for d in PLOT_D:

    sub = (
        df[
            df["d"] == d
        ]
        .sort_values("p")
        .copy()
    )

    w_pred = (
        predicted_w0(d)
    )

    if d not in FIT_D:

        print(
            f"d={d:3d}: "
            f"fit omitted "
            f"(pre-asymptotic tail; "
            f"w0={w_pred})"
        )

        fit_rows.append({
            "d": d,
            "N": 0,
            "p_min": np.nan,
            "p_max": np.nan,
            "A": np.nan,
            "lambda": np.nan,
            "lambda_err": np.nan,
            "w0_pred": w_pred,
            "chi2_red": np.nan,
            "status": "pre-asymptotic",
        })

        continue


    n_fit = (
        FIT_N_LOWEST[d]
    )


    fit = fit_lowest_points(
        sub,
        n_fit,
    )


    if fit is None:

        print(
            f"d={d:3d}: "
            f"insufficient resolved points"
        )

        continue


    fits[d] = fit


    print(
        f"d={d:3d}  "
        f"N={fit['N']:2d}  "
        f"p=["
        f"{fit['p_min']:.6f},"
        f"{fit['p_max']:.6f}"
        f"]  "
        f"lambda="
        f"{fit['lambda']:.4f} "
        f"+/- "
        f"{fit['lambda_err']:.4f}  "
        f"w0={w_pred:2d}  "
        f"chi2_red="
        f"{fit['chi2_red']:.3f}"
    )


    fit_rows.append({
        "d": d,
        "N": fit["N"],
        "p_min": fit["p_min"],
        "p_max": fit["p_max"],
        "A": fit["A"],
        "lambda": fit["lambda"],
        "lambda_err": fit[
            "lambda_err"
        ],
        "w0_pred": w_pred,
        "chi2_red": fit[
            "chi2_red"
        ],
        "status": "fit",
    })


# =============================================================================
# Save fit table
# =============================================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


pd.DataFrame(
    fit_rows
).to_csv(
    OUTPUT_FITS,
    index=False,
)


# =============================================================================
# Main figure
# =============================================================================

fig, ax = plt.subplots(
    figsize=(
        5,
        4,
    )
)


# =============================================================================
# Monte Carlo curves
# =============================================================================
#
# Only reduced points are shown.
#
# Fits still use the complete resolved low-p data.
# =============================================================================

for d in PLOT_D:

    sub_full = (
        df[
            df["d"] == d
        ]
        .sort_values("p")
        .copy()
    )


    if sub_full.empty:
        continue


    sub = reduced_plot_subset(
        sub_full,
        MAX_PLOT_POINTS,
    )


    ax.errorbar(
        sub["p"],
        sub["pL"],
        yerr=sub["pL_se"],
        linestyle="-",
        marker="o",
        markerfacecolor="none",
        capsize=2,
        label=str(d),
        zorder=2,
    )


# =============================================================================
# Low-p fits
# =============================================================================

for d in FIT_D:

    if d not in fits:
        continue


    fit = (
        fits[d]
    )


    p_fit = np.logspace(
        np.log10(
            fit["p_min"]
        ),
        np.log10(
            fit["p_max"]
        ),
        300,
    )


    y_fit = power_law(
        p_fit,
        fit["A"],
        fit["lambda"],
    )


    ax.plot(
        p_fit,
        y_fit,
        "--",
        color="black",
        linewidth=1.4,
        zorder=4,
    )


# =============================================================================
# Axes
# =============================================================================

ax.set_xscale(
    "log"
)

ax.set_yscale(
    "log"
)


ax.set_xlim(
    XMIN,
    XMAX,
)

ax.set_ylim(
    YMIN,
    YMAX,
)


ax.set_xlabel(
    r"$p=q$"
)

ax.set_ylabel(
    r"$p_L$"
)


# =============================================================================
# Logarithmic ticks
# =============================================================================

ax.xaxis.set_major_locator(
    LogLocator(
        base=10.0,
        subs=(1.0,),
        numticks=10,
    )
)

ax.xaxis.set_major_formatter(
    LogFormatterMathtext(
        base=10.0
    )
)

ax.xaxis.set_minor_locator(
    LogLocator(
        base=10.0,
        subs=np.arange(
            2,
            10,
        ) * 0.1,
        numticks=100,
    )
)

ax.xaxis.set_minor_formatter(
    NullFormatter()
)


ax.yaxis.set_major_locator(
    LogLocator(
        base=10.0,
        subs=(1.0,),
        numticks=20,
    )
)

ax.yaxis.set_major_formatter(
    LogFormatterMathtext(
        base=10.0
    )
)

ax.yaxis.set_minor_locator(
    LogLocator(
        base=10.0,
        subs=np.arange(
            2,
            10,
        ) * 0.1,
        numticks=100,
    )
)

ax.yaxis.set_minor_formatter(
    NullFormatter()
)


# =============================================================================
# Grid
# =============================================================================

ax.grid(
    True,
    which="major",
)

# =============================================================================
# Legend above plot
# =============================================================================

ax.legend(
    title=r"$d$",
    bbox_to_anchor=(
        0,
        1.02,
        1,
        0.20,
    ),
    loc="lower left",
    mode="expand",
    borderaxespad=0,
    ncol=5,
)


# =============================================================================
# Inset
# =============================================================================
#
# Same visual logic as the code-capacity figure:
#
#   purple horizontal staircase segments:
#       exact discrete prediction w0(d)
#
#   black open circles:
#       fitted phenomenological exponents lambda(d)
#
# =============================================================================

ax2 = fig.add_axes([
    0.61,   # left
    0.22,   # bottom
    0.27,   # width
    0.27,   # height
])


# =============================================================================
# Exact discrete staircase prediction
# =============================================================================

odd_ds = np.arange(
    1,
    INSET_D_MAX + 1,
    2,
    dtype=int,
)


for d in odd_ds:

    w = predicted_w0(
        int(d)
    )


    ax2.hlines(
        y=w,
        xmin=max(
            0,
            d - 1,
        ),
        xmax=d + 1,
        color="purple",
        linewidth=2.0,
        zorder=1,
    )


# =============================================================================
# Numerical fitted exponents
# =============================================================================

fit_ds = []
fit_lam = []
fit_err = []


for d in PLOT_D:

    if d not in fits:
        continue

    fit_ds.append(
        d
    )

    fit_lam.append(
        fits[d][
            "lambda"
        ]
    )

    fit_err.append(
        fits[d][
            "lambda_err"
        ]
    )


if fit_ds:

    ax2.errorbar(
        fit_ds,
        fit_lam,
        yerr=fit_err,
        linestyle="none",
        marker="o",
        markersize=5,
        markerfacecolor="none",
        color="black",
        ecolor="black",
        capsize=2,
        zorder=3,
    )


# =============================================================================
# Inset formatting
# =============================================================================

ax2.set_xlim(
    0,
    INSET_D_MAX,
)


max_pred = max(
    predicted_w0(
        int(d)
    )
    for d in odd_ds
)


max_fit = (
    max(
        fit_lam
    )
    if fit_lam
    else 0.0
)


ax2.set_ylim(
    0,
    1.10 * max(
        max_pred,
        max_fit,
    ),
)


ax2.set_xlabel(
    r"$d$",
    fontsize=13,
    labelpad=1,
)

ax2.set_ylabel(
    r"$\lambda$",
    fontsize=13,
    labelpad=1,
)


ax2.tick_params(
    axis="both",
    which="major",
    labelsize=9,
)

ax2.minorticks_on()


# =============================================================================
# Save
# =============================================================================

fig.savefig(
    OUTPUT_PDF,
    bbox_inches="tight",
)

fig.savefig(
    OUTPUT_PNG,
    dpi=300,
    bbox_inches="tight",
)


plt.show()


print()
print("=" * 108)

print(
    f"Saved: "
    f"{OUTPUT_PDF}"
)

print(
    f"Saved: "
    f"{OUTPUT_PNG}"
)

print(
    f"Saved: "
    f"{OUTPUT_FITS}"
)

print("=" * 108)
