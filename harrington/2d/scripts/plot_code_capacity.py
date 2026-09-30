# scripts/plot_code_capacity.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

INPUT = Path("data/code_capacity/code_capacity.csv")

OUTPUT_DIR = Path("figs")
OUTPUT_PDF = OUTPUT_DIR / "har2d_cc.pdf"
OUTPUT_PNG = OUTPUT_DIR / "har2d_cc.png"
OUTPUT_FITS = OUTPUT_DIR / "har2d_cc_tail_fits.csv"

PLOT_D = [3, 9, 27, 81]

# Numerical power-law fits are shown ONLY where the sampled p-range
# has reached the asymptotic minimum-weight regime.
FIT_WINDOWS = {
    3: (0.0100, 0.0350),
    9: (0.0025, 0.0150),
}

# Distances intentionally omitted from numerical tail fitting because
# the accessible Monte Carlo window remains pre-asymptotic.
PREASYMPTOTIC_D = [27, 81]

# Harrington2D code-capacity threshold from FSS.
PC = 0.04523
PC_ERR = 0.00002

XMIN = 1e-2
XMAX = 0.20

YMIN = 1e-6
YMAX = 1.0

MIN_FIT_POINTS = 4


# Plot a reasonably sparse set of points while retaining the low-p tails.
PLOT_P = np.array([
    0.0025,
    0.00375,
    0.0050,
    0.0075,

    0.0100,
    0.0125,
    0.0150,
    0.0175,
    0.0200,

    0.0250,
    0.0300,
    0.0350,
    0.0400,
    0.0450,
    0.0500,
    0.0550,
    0.0600,

    0.0700,
    0.0800,
    0.1000,
    0.1200,
    0.1500,
    0.1800,
    0.2000,
])


# =============================================================================
# Style
# =============================================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
})


# =============================================================================
# Theory
# =============================================================================

def harrington_lambda(d):
    """
    Expected asymptotic Harrington hierarchy exponent

        lambda(d) = 2^(log_3 d)
                  = d^(log_3 2).
    """
    d = np.asarray(d, dtype=float)

    return d ** (
        np.log(2.0)
        / np.log(3.0)
    )


# =============================================================================
# Fit helpers
# =============================================================================

def weighted_power_law_fit(sub, pmin, pmax):

    s = sub[
        (sub["p"] >= pmin)
        & (sub["p"] <= pmax)
        & np.isfinite(sub["pL"])
        & np.isfinite(sub["se"])
        & (sub["pL"] > 0)
        & (sub["se"] > 0)
    ].sort_values("p")

    if len(s) < MIN_FIT_POINTS:
        return None

    p = s["p"].to_numpy(float)
    pL = s["pL"].to_numpy(float)
    se = s["se"].to_numpy(float)

    x = np.log(p)
    y = np.log(pL)

    sigma = se / pL
    w = 1.0 / sigma**2

    X = np.column_stack([
        np.ones_like(x),
        x,
    ])

    XT_W = X.T * w

    cov = np.linalg.pinv(
        XT_W @ X
    )

    beta = (
        cov
        @ XT_W
        @ y
    )

    logA, lam = beta

    yfit = X @ beta

    chi2 = np.sum(
        ((y - yfit) / sigma)**2
    )

    dof = len(y) - 2

    return {
        "A": np.exp(logA),
        "lambda": lam,
        "lambda_err": np.sqrt(cov[1, 1]),
        "chi2_red": (
            chi2 / dof
            if dof > 0
            else np.nan
        ),
        "N": len(s),
        "data": s,
    }


def power_law(p, A, lam):
    return A * np.asarray(p, float)**lam


def is_plot_p(p):
    return np.any(
        np.isclose(
            p,
            PLOT_P,
            rtol=0,
            atol=1e-10,
        )
    )


# =============================================================================
# Load
# =============================================================================

df = pd.read_csv(INPUT)

required = {
    "d",
    "p",
    "pL",
    "se",
}

missing = required - set(df.columns)

if missing:
    raise RuntimeError(
        f"Missing columns: {sorted(missing)}"
    )

df = df[
    df["d"].isin(PLOT_D)
    & np.isfinite(df["p"])
    & np.isfinite(df["pL"])
    & np.isfinite(df["se"])
    & (df["p"] > 0)
    & (df["pL"] > 0)
    & (df["se"] > 0)
].copy()


# =============================================================================
# Plot subset
# =============================================================================

df_plot = df[
    df["p"].apply(is_plot_p)
].copy()


# Always retain the lowest two resolved p-points for each distance,
# even if they are not explicitly in PLOT_P.
lowest = (
    df
    .sort_values(["d", "p"])
    .groupby(
        "d",
        group_keys=False,
    )
    .head(2)
)

df_plot = (
    pd.concat(
        [df_plot, lowest],
        ignore_index=True,
    )
    .drop_duplicates(
        subset=["d", "p"]
    )
    .sort_values(["d", "p"])
)


# =============================================================================
# Fits
# =============================================================================

fits = {}
fit_rows = []

print()
print("=" * 88)
print("HARRINGTON2D CODE-CAPACITY ASYMPTOTIC POWER-LAW FITS")
print("=" * 88)

for d in PLOT_D:

    lam_th = float(
        harrington_lambda(d)
    )

    if d not in FIT_WINDOWS:

        print(
            f"d={d:3d}: "
            f"no numerical fit shown; "
            f"accessible p-range is pre-asymptotic. "
            f"Expected w_min = {lam_th:.6f}"
        )

        fit_rows.append({
            "d": d,
            "p_min": np.nan,
            "p_max": np.nan,
            "lambda": np.nan,
            "lambda_err": np.nan,
            "lambda_theory": lam_th,
            "N": 0,
            "chi2_red": np.nan,
            "status": "pre-asymptotic",
        })

        continue

    pmin, pmax = FIT_WINDOWS[d]

    sub = df[
        df["d"] == d
    ]

    fit = weighted_power_law_fit(
        sub,
        pmin,
        pmax,
    )

    if fit is None:

        print(
            f"d={d:3d}: insufficient points"
        )

        continue

    fits[d] = fit

    print(
        f"d={d:3d}  "
        f"window=[{pmin:.4f},{pmax:.4f}]  "
        f"lambda={fit['lambda']:.6f} "
        f"+/- {fit['lambda_err']:.6f}  "
        f"lambda_theory={lam_th:.6f}  "
        f"N={fit['N']:2d}  "
        f"chi2_red={fit['chi2_red']:.3f}"
    )

    fit_rows.append({
        "d": d,
        "p_min": pmin,
        "p_max": pmax,
        "lambda": fit["lambda"],
        "lambda_err": fit["lambda_err"],
        "lambda_theory": lam_th,
        "N": fit["N"],
        "chi2_red": fit["chi2_red"],
        "status": "fit",
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

fig = plt.figure(
    figsize=(5, 4)
)

ax = plt.gca()


# -----------------------------------------------------------------------------
# Monte Carlo curves
# -----------------------------------------------------------------------------

for d in PLOT_D:

    sub = (
        df_plot[
            df_plot["d"] == d
        ]
        .sort_values("p")
    )

    if sub.empty:
        continue

    ax.errorbar(
        sub["p"],
        sub["pL"],
        yerr=sub["se"],
        linestyle="-",
        marker="o",
        markerfacecolor="none",
        capsize=2,
        label=str(d),
        zorder=2,
    )


# -----------------------------------------------------------------------------
# Asymptotic numerical fits: d=3 and d=9 only
# -----------------------------------------------------------------------------

for d, fit in fits.items():

    pmin, pmax = FIT_WINDOWS[d]

    # Extend the fitted asymptotic law toward smaller p,
    # as in the other code-capacity figures.
    p_fit = np.logspace(
        np.log10(XMIN),
        np.log10(pmax),
        400,
    )

    pL_fit = power_law(
        p_fit,
        fit["A"],
        fit["lambda"],
    )

    visible = (
        (pL_fit >= YMIN)
        & (pL_fit <= YMAX)
    )

    ax.plot(
        p_fit[visible],
        pL_fit[visible],
        "--",
        color="black",
        linewidth=1.5,
        zorder=4,
    )


# -----------------------------------------------------------------------------
# Threshold
# -----------------------------------------------------------------------------

# The uncertainty is so narrow that it is barely visible, but keep it for
# consistency with the quoted FSS estimate.
ax.axvspan(
    PC - PC_ERR,
    PC + PC_ERR,
    color="red",
    alpha=0.08,
    zorder=0,
)

ax.axvline(
    PC,
    linestyle="--",
    color="red",
    linewidth=1.2,
    zorder=3,
)


# -----------------------------------------------------------------------------
# Axes
# -----------------------------------------------------------------------------

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
    r"$p$"
)

ax.set_ylabel(
    r"$p_L$"
)

ax.grid()


# -----------------------------------------------------------------------------
# Legend
# -----------------------------------------------------------------------------

ax.legend(
    title=r"$d$",
    bbox_to_anchor=(
        0,
        1.02,
        1,
        0.2,
    ),
    loc="lower left",
    mode="expand",
    borderaxespad=0,
    ncol=4,
)


# =============================================================================
# Inset
# =============================================================================

left = 0.64
bottom = 0.22
width = 0.24
height = 0.30

ax2 = fig.add_axes([
    left,
    bottom,
    width,
    height,
])


# -----------------------------------------------------------------------------
# Expected asymptotic hierarchy
# -----------------------------------------------------------------------------

d_theory = np.linspace(
    1,
    max(PLOT_D),
    500,
)

ax2.plot(
    d_theory,
    harrington_lambda(
        d_theory
    ),
    color="purple",
    linewidth=1.5,
    zorder=2,
)


# -----------------------------------------------------------------------------
# Only numerical fits which have reached the asymptotic regime
# -----------------------------------------------------------------------------

fit_ds = []
fit_lam = []
fit_err = []

for d in sorted(fits):

    fit_ds.append(d)
    fit_lam.append(
        fits[d]["lambda"]
    )
    fit_err.append(
        fits[d]["lambda_err"]
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
        capsize=2,
        zorder=3,
    )


ax2.set_xlabel(
    r"$d$",
    fontsize=14,
    labelpad=0,
)

ax2.set_ylabel(
    r"$\lambda$",
    fontsize=14,
)

ax2.tick_params(
    axis="both",
    which="major",
    labelsize=10,
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
print("=" * 88)
print(f"Saved: {OUTPUT_PDF}")
print(f"Saved: {OUTPUT_PNG}")
print(f"Saved: {OUTPUT_FITS}")
print("=" * 88)
