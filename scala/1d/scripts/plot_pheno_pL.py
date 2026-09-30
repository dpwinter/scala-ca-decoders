# scripts/plot_pheno_pL.py

from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter
import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

INPUT = Path("data/pheno/pheno_pL_fixed.csv")

OUTPUT_DIR = Path("figs")
OUTPUT_PDF = OUTPUT_DIR / "scala1d_pheno_pL.pdf"
OUTPUT_PNG = OUTPUT_DIR / "scala1d_pheno_pL.png"
OUTPUT_FITS = OUTPUT_DIR / "scala1d_pheno_tail_fits.csv"


# Same distances as SCALA2D code-capacity plot
PLOT_D = [
    3,
    9,
    15,
    21,
    31,
    51,
    81,
    111,
]


# Fit only distances for which we expect the currently accessible
# low-p region to support an asymptotic tail fit.
FIT_D = [
    3,
    9,
    15,
    21,
    31,
]


# IMPORTANT:
# Fit the N lowest resolved points, not a fixed p interval.
#
# This makes "tail fit" mean exactly what it says.
FIT_N_LOWEST = {
    3:  6,
    9:  6,
    15: 5,
    21: 5,
    31: 5,
    51: 4,
}


XMIN = 1e-2
XMAX = 1e-1

YMIN = 1e-8
YMAX = 1e-1

INSET_D_MAX = 120


plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
})


# =============================================================================
# Exact discrete SCALA failure-weight prediction
# =============================================================================

def min_factor_sum(w):
    """
    f(w) = min{k+n : k,n positive integers and k*n = w}.
    """
    best = w + 1

    for k in range(1, math.isqrt(w) + 1):
        if w % k == 0:
            n = w // k
            best = min(best, k + n)

    return best


def predicted_w0(d):
    """
    Exact integer prediction of Eq. (2):

        w0(d) = min { w in Z_+ :
                      2w - f(w) + 1 >= (d+1)/2 }.
    """
    target = (d + 1) / 2.0

    w = 1

    while True:
        if 2 * w - min_factor_sum(w) + 1 >= target:
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
        .isin(["true", "1", "yes"])
    )


def weighted_linear_fit(X, y, sigma):
    weights = 1.0 / sigma**2

    XT_W = X.T * weights
    cov = np.linalg.pinv(XT_W @ X)

    beta = cov @ XT_W @ y

    residual = y - X @ beta

    chi2 = np.sum((residual / sigma)**2)

    dof = len(y) - X.shape[1]

    chi2_red = (
        chi2 / dof
        if dof > 0
        else np.nan
    )

    return beta, cov, chi2, chi2_red


def fit_lowest_points(sub, n_points):
    """
    Fit the n_points lowest-p resolved data points to

        pL = A p^lambda.

    Fit is performed in log space with propagated MC errors.
    """

    sub = (
        sub[
            (sub["p"] > 0)
            & (sub["pL"] > 0)
            & (sub["pL_se"] > 0)
            & np.isfinite(sub["p"])
            & np.isfinite(sub["pL"])
            & np.isfinite(sub["pL_se"])
        ]
        .sort_values("p")
        .copy()
    )

    if len(sub) < n_points:
        return None

    fit_sub = (
        sub.iloc[:n_points]
        .copy()
    )

    p = fit_sub["p"].to_numpy(float)
    pL = fit_sub["pL"].to_numpy(float)
    se = fit_sub["pL_se"].to_numpy(float)

    x = np.log(p)
    y = np.log(pL)

    sigma_log = se / pL

    X = np.column_stack([
        np.ones_like(x),
        x,
    ])

    beta, cov, chi2, chi2_red = weighted_linear_fit(
        X,
        y,
        sigma_log,
    )

    log_A, lam = beta

    return {
        "A": float(np.exp(log_A)),
        "lambda": float(lam),
        "lambda_err": float(np.sqrt(cov[1, 1])),
        "chi2": float(chi2),
        "chi2_red": float(chi2_red),
        "N": len(fit_sub),
        "data": fit_sub,
        "p_min": float(p.min()),
        "p_max": float(p.max()),
    }


def power_law(p, A, lam):
    return A * np.asarray(p, dtype=float)**lam


# =============================================================================
# Load data
# =============================================================================

df = pd.read_csv(INPUT)


required = {
    "d",
    "p",
    "q",
    "stationary",
    "pL",
    "pL_se",
}


missing = required - set(df.columns)

if missing:
    raise RuntimeError(
        f"Missing columns: {sorted(missing)}"
    )


stationary = as_bool(df["stationary"])


df = (
    df[
        stationary
        & df["d"].isin(PLOT_D)
        & np.isclose(
            df["p"],
            df["q"],
            atol=1e-12,
            rtol=0.0,
        )
        & (df["p"] > 0.0)
        & (df["pL"] > 0.0)
        & (df["pL_se"] > 0.0)
        & np.isfinite(df["p"])
        & np.isfinite(df["pL"])
        & np.isfinite(df["pL_se"])
    ]
    .copy()
)


# =============================================================================
# Fits
# =============================================================================

fits = {}
fit_rows = []


print()
print("=" * 105)
print("SCALA1D phenomenological low-p tail fits")
print("=" * 105)


for d in PLOT_D:

    sub = (
        df[df["d"] == d]
        .sort_values("p")
        .copy()
    )

    if sub.empty:
        print(
            f"d={d:3d}: no data"
        )
        continue


    if d not in FIT_D:
        print(
            f"d={d:3d}: fit omitted"
        )
        continue


    n_fit = FIT_N_LOWEST[d]

    fit = fit_lowest_points(
        sub,
        n_fit,
    )


    if fit is None:
        print(
            f"d={d:3d}: only {len(sub)} resolved points; "
            f"need {n_fit}"
        )
        continue


    fits[d] = fit

    w_pred = predicted_w0(d)


    print(
        f"d={d:3d}  "
        f"N={fit['N']:2d}  "
        f"p=[{fit['p_min']:.5f},{fit['p_max']:.5f}]  "
        f"lambda={fit['lambda']:.5f} "
        f"+/- {fit['lambda_err']:.5f}  "
        f"w0={w_pred:2d}  "
        f"chi2_red={fit['chi2_red']:.3f}"
    )


    fit_rows.append({
        "d": d,
        "N": fit["N"],
        "p_min": fit["p_min"],
        "p_max": fit["p_max"],
        "A": fit["A"],
        "lambda": fit["lambda"],
        "lambda_err": fit["lambda_err"],
        "w0_pred": w_pred,
        "chi2_red": fit["chi2_red"],
    })


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
    figsize=(5, 4)
)


# =============================================================================
# Monte Carlo curves
# =============================================================================

for d in PLOT_D:

    sub = (
        df[df["d"] == d]
        .sort_values("p")
        .copy()
    )

    if sub.empty:
        continue


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
# Tail fits
# =============================================================================

for d, fit in fits.items():

    # ONLY draw over the fitted tail interval.
    #
    # Do not extrapolate the dashed line into the visibly curved regime.
    p_fit = np.logspace(
        np.log10(fit["p_min"]),
        np.log10(fit["p_max"]),
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
# Main-axis formatting
# =============================================================================

ax.set_xscale("log")
ax.set_yscale("log")

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


# Only show 10^-2 and 10^-1 as x labels.
ax.xaxis.set_major_locator(
    LogLocator(
        base=10.0,
        subs=(1.0,),
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
        subs=np.arange(2, 10) * 0.1,
    )
)

ax.xaxis.set_minor_formatter(
    NullFormatter()
)


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
    ncol=4,
)


# =============================================================================
# Inset
# =============================================================================
#
# Moved upward relative to previous version so that its d label no longer
# collides visually with the main x-axis label.
# =============================================================================

ax2 = fig.add_axes([
    0.61,   # left
    0.235,  # bottom -- raised from ~0.20
    0.27,   # width
    0.27,   # height
])


# =============================================================================
# Exact staircase prediction
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
        xmin=max(0, d - 1),
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

    fit_ds.append(d)

    fit_lam.append(
        fits[d]["lambda"]
    )

    fit_err.append(
        fits[d]["lambda_err"]
    )


ax2.errorbar(
    fit_ds,
    fit_lam,
    yerr=fit_err,
    linestyle="none",
    marker="o",
    markersize=5,
    markerfacecolor="none",
    color="black",
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
    predicted_w0(int(d))
    for d in odd_ds
)

max_fit = (
    max(fit_lam)
    if fit_lam
    else 0.0
)


ax2.set_ylim(
    0,
    1.07 * max(
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
print("=" * 105)
print(f"Saved: {OUTPUT_PDF}")
print(f"Saved: {OUTPUT_PNG}")
print(f"Saved: {OUTPUT_FITS}")
print("=" * 105)
