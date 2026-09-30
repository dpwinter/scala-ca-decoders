# scripts/plot_harrington2d_pheno_matched.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Input / output
# =============================================================================

BASELINE_CSV = Path(
    "data/pheno/harrington2d_pheno_matched.csv"
)

SIGNAL_CSV = Path(
    "data/pheno/harrington2d_pheno_signal_full.csv"
)

OUTPUT_DIR = Path("figs")

OUT_SIGNAL = (
    OUTPUT_DIR
    / "harrington2d_pheno_signal_matched.pdf"
)

OUT_SIGNAL_PNG = (
    OUTPUT_DIR
    / "harrington2d_pheno_signal_matched.png"
)

OUT_PHENO = (
    OUTPUT_DIR
    / "harrington2d_pheno_matched.pdf"
)

OUT_PHENO_PNG = (
    OUTPUT_DIR
    / "harrington2d_pheno_matched.png"
)

OUT_FITS = (
    OUTPUT_DIR
    / "harrington2d_pheno_matched_fits.csv"
)


# =============================================================================
# Plot configuration
# =============================================================================

DS = [
    3,
    9,
    27,
    81,
]

COLORS = {
    3: "tab:blue",
    9: "tab:orange",
    27: "tab:green",
    81: "tab:red",
}


# =============================================================================
# Fit configuration
# =============================================================================

FIT_DISTANCES = [
    3,
    9,
]

FIT_WINDOWS = {
    3: (
        1.0e-3,
        4.0e-3,
    ),
    9: (
        1.0e-3,
        3.0e-3,
    ),
}

MIN_FIT_POINTS = 3


# =============================================================================
# Axes
# =============================================================================

XMIN = 1.0e-3
XMAX = 1.0

YMIN = 1.0
YMAX = 1.0e7


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 14,
        "legend.fontsize": 11,
    }
)


# =============================================================================
# Theory
# =============================================================================

def lambda_theory(d):
    """
    Harrington hierarchy prediction:

        lambda(d) = d^(log_3 2)
    """

    d = np.asarray(
        d,
        dtype=float,
    )

    return (
        d
        ** (
            np.log(2.0)
            / np.log(3.0)
        )
    )


# =============================================================================
# Weighted log-log fit
# =============================================================================

def weighted_log_fit(
    sub,
    rmin,
    rmax,
):

    s = sub[
        (sub["r"] >= rmin)
        & (sub["r"] <= rmax)
        & np.isfinite(
            sub["r"]
        )
        & np.isfinite(
            sub["mean_TF"]
        )
        & np.isfinite(
            sub["se_TF"]
        )
        & (
            sub["r"] > 0
        )
        & (
            sub["mean_TF"] > 0
        )
        & (
            sub["se_TF"] > 0
        )
    ].copy()

    s = s.sort_values(
        "r"
    )

    if len(s) < MIN_FIT_POINTS:
        return None

    r = s[
        "r"
    ].to_numpy(
        dtype=float
    )

    T = s[
        "mean_TF"
    ].to_numpy(
        dtype=float
    )

    T_se = s[
        "se_TF"
    ].to_numpy(
        dtype=float
    )

    x = np.log(
        r
    )

    y = np.log(
        T
    )

    sigma_y = (
        T_se
        / T
    )

    w = (
        1.0
        / sigma_y**2
    )

    X = np.column_stack(
        [
            np.ones_like(
                x
            ),
            x,
        ]
    )

    XT_W = (
        X.T
        * w
    )

    covariance = np.linalg.pinv(
        XT_W
        @ X
    )

    beta = (
        covariance
        @ XT_W
        @ y
    )

    log_A = float(
        beta[0]
    )

    slope = float(
        beta[1]
    )

    A = float(
        np.exp(
            log_A
        )
    )

    lam = float(
        -slope
    )

    lam_se = float(
        np.sqrt(
            covariance[
                1,
                1,
            ]
        )
    )

    y_fit = (
        X
        @ beta
    )

    chi2 = float(
        np.sum(
            (
                (
                    y
                    - y_fit
                )
                / sigma_y
            )
            ** 2
        )
    )

    dof = (
        len(y)
        - 2
    )

    if dof > 0:

        chi2_red = (
            chi2
            / dof
        )

    else:

        chi2_red = np.nan

    return {
        "A": A,
        "lambda": lam,
        "lambda_se": lam_se,
        "chi2_red": chi2_red,
        "N": len(s),
        "data": s,
    }


def lifetime_power_law(
    r,
    A,
    lam,
):

    return (
        A
        * np.asarray(
            r,
            dtype=float,
        )
        ** (
            -lam
        )
    )


# =============================================================================
# Data loading helper
# =============================================================================

def load_resolved_csv(
    path,
):

    df = pd.read_csv(
        path
    )

    required = {
        "model",
        "d",
        "r",
        "mean_TF",
        "se_TF",
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:

        raise RuntimeError(
            f"Missing columns in {path}: "
            f"{sorted(missing)}"
        )


    if "resolved" in df.columns:

        if (
            df[
                "resolved"
            ].dtype
            == object
        ):

            resolved = (
                df[
                    "resolved"
                ]
                .astype(str)
                .str.lower()
                .isin(
                    [
                        "true",
                        "1",
                    ]
                )
            )

        else:

            resolved = (
                df[
                    "resolved"
                ]
                .astype(bool)
            )

        df = (
            df.loc[
                resolved
            ]
            .copy()
        )


    df = df[
        np.isfinite(
            df["r"]
        )
        & np.isfinite(
            df["mean_TF"]
        )
        & np.isfinite(
            df["se_TF"]
        )
        & (
            df["r"] > 0
        )
        & (
            df["mean_TF"] > 0
        )
        & (
            df["se_TF"] > 0
        )
    ].copy()


    return df


# =============================================================================
# Load baseline and signal data separately
# =============================================================================

baseline_df = load_resolved_csv(
    BASELINE_CSV
)

signal_df = load_resolved_csv(
    SIGNAL_CSV
)


# =============================================================================
# Curve helpers
# =============================================================================

def get_baseline_curve(
    d,
):

    x = baseline_df[
        (
            baseline_df["model"]
            == "baseline"
        )
        & (
            baseline_df["d"]
            == d
        )
    ].copy()

    return x.sort_values(
        "r"
    )


def get_signal_curve(
    d,
):

    x = signal_df[
        (
            signal_df["model"]
            == "full"
        )
        & (
            signal_df["d"]
            == d
        )
    ].copy()

    return x.sort_values(
        "r"
    )


# =============================================================================
# Fit baseline curves
# =============================================================================

fits = {}

fit_rows = []


print()
print(
    "=" * 88
)

print(
    "HARRINGTON2D MATCHED PHENOMENOLOGICAL "
    "ASYMPTOTIC POWER-LAW FITS"
)

print(
    "=" * 88
)


baseline = baseline_df[
    baseline_df["model"]
    == "baseline"
].copy()


for d in DS:

    lam_th = float(
        lambda_theory(
            d
        )
    )

    if d not in FIT_DISTANCES:

        print(
            f"d={d:3d}: "
            "no numerical fit shown; "
            "accessible low-r regime treated as "
            "pre-asymptotic. "
            f"lambda_theory={lam_th:.6f}"
        )

        fit_rows.append(
            {
                "d": d,
                "r_min": np.nan,
                "r_max": np.nan,
                "lambda": np.nan,
                "lambda_se": np.nan,
                "lambda_theory": lam_th,
                "N": 0,
                "chi2_red": np.nan,
                "status": "pre-asymptotic",
            }
        )

        continue


    rmin, rmax = (
        FIT_WINDOWS[
            d
        ]
    )

    sub = baseline[
        baseline["d"]
        == d
    ]

    fit = weighted_log_fit(
        sub,
        rmin,
        rmax,
    )

    if fit is None:

        print(
            f"d={d:3d}: "
            "insufficient resolved points "
            f"in [{rmin:g}, {rmax:g}]"
        )

        fit_rows.append(
            {
                "d": d,
                "r_min": rmin,
                "r_max": rmax,
                "lambda": np.nan,
                "lambda_se": np.nan,
                "lambda_theory": lam_th,
                "N": 0,
                "chi2_red": np.nan,
                "status": "insufficient",
            }
        )

        continue


    fits[
        d
    ] = fit


    print(
        f"d={d:3d}  "
        f"window=[{rmin:.4g},{rmax:.4g}]  "
        f"lambda={fit['lambda']:.6f} "
        f"+/- {fit['lambda_se']:.6f}  "
        f"lambda_theory={lam_th:.6f}  "
        f"N={fit['N']:2d}  "
        f"chi2_red={fit['chi2_red']:.3f}"
    )


    fit_rows.append(
        {
            "d": d,
            "r_min": rmin,
            "r_max": rmax,
            "lambda": fit[
                "lambda"
            ],
            "lambda_se": fit[
                "lambda_se"
            ],
            "lambda_theory": lam_th,
            "N": fit[
                "N"
            ],
            "chi2_red": fit[
                "chi2_red"
            ],
            "status": "fit",
        }
    )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


pd.DataFrame(
    fit_rows
).to_csv(
    OUT_FITS,
    index=False,
)


# =============================================================================
# Shared formatting
# =============================================================================

def format_axis(
    ax,
    xlabel,
):

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
        xlabel
    )

    ax.set_ylabel(
        r"$\langle T_F\rangle$"
    )

    ax.grid(
        True,
        which="major",
        alpha=0.6,
    )


def add_code_capacity_style_legend(
    ax,
):

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
# Figure 1:
#
# Signal matched
#
# faint:
#     p=q
#
# solid:
#     p=q=p_count=p_flip
#
# Legend INSIDE plot.
# =============================================================================

fig = plt.figure(
    figsize=(5, 4)
)

ax = plt.gca()


for d in DS:

    color = COLORS[
        d
    ]


    # -------------------------------------------------------------------------
    # Baseline p=q
    # -------------------------------------------------------------------------

    sub = get_baseline_curve(
        d
    )

    if not sub.empty:

        ax.errorbar(
            sub["r"],
            sub["mean_TF"],
            yerr=sub["se_TF"],
            color=color,
            linestyle="-",
            linewidth=1.5,
            marker="o",
            markersize=5,
            markerfacecolor="none",
            markeredgewidth=1.2,
            elinewidth=1.0,
            capsize=2.0,
            capthick=1.0,
            alpha=0.25,
            zorder=2,
        )


    # -------------------------------------------------------------------------
    # Full signal noise from SIGNAL_CSV
    # -------------------------------------------------------------------------

    sub = get_signal_curve(
        d
    )

    if not sub.empty:

        ax.errorbar(
            sub["r"],
            sub["mean_TF"],
            yerr=sub["se_TF"],
            color=color,
            linestyle="-",
            linewidth=1.5,
            marker="o",
            markersize=5,
            markerfacecolor="none",
            markeredgewidth=1.2,
            elinewidth=1.0,
            capsize=2.0,
            capthick=1.0,
            zorder=4,
        )


format_axis(
    ax,
    r"$p=q=p_{\rm sig}$",
)


# -----------------------------------------------------------------------------
# Distance legend inside plot
# -----------------------------------------------------------------------------

handles = []

for d in DS:

    h, = ax.plot(
        [],
        [],
        color=COLORS[
            d
        ],
        linestyle="-",
        linewidth=1.5,
        marker="o",
        markersize=5,
        markerfacecolor="none",
        label=str(
            d
        ),
    )

    handles.append(
        h
    )


ax.legend(
    handles=handles,
    title=r"$d$",
    loc="upper right",
    ncol=1,
    frameon=True,
)


fig.tight_layout()


fig.savefig(
    OUT_SIGNAL,
    bbox_inches="tight",
)

fig.savefig(
    OUT_SIGNAL_PNG,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


print(
    f"Saved: {OUT_SIGNAL}"
)


# =============================================================================
# Figure 2:
#
# Baseline matched phenomenological noise:
#
#     p=q=r
#
# Legend above + inset
# =============================================================================

fig = plt.figure(
    figsize=(5, 4)
)

ax = plt.gca()


# -----------------------------------------------------------------------------
# Monte Carlo curves
# -----------------------------------------------------------------------------

for d in DS:

    sub = get_baseline_curve(
        d
    )

    if sub.empty:
        continue

    ax.errorbar(
        sub["r"],
        sub["mean_TF"],
        yerr=sub["se_TF"],
        color=COLORS[
            d
        ],
        linestyle="-",
        linewidth=1.5,
        marker="o",
        markersize=5,
        markerfacecolor="none",
        markeredgewidth=1.2,
        elinewidth=1.0,
        capsize=2.0,
        capthick=1.0,
        label=str(
            d
        ),
        zorder=2,
    )


# -----------------------------------------------------------------------------
# Numerical asymptotic fits
# -----------------------------------------------------------------------------

for d in FIT_DISTANCES:

    if d not in fits:
        continue

    fit = fits[
        d
    ]

    _, rmax = (
        FIT_WINDOWS[
            d
        ]
    )

    r_fit = np.logspace(
        np.log10(
            XMIN
        ),
        np.log10(
            rmax
        ),
        300,
    )

    T_fit = lifetime_power_law(
        r_fit,
        fit["A"],
        fit["lambda"],
    )

    visible = (
        (T_fit >= YMIN)
        & (T_fit <= YMAX)
    )

    ax.plot(
        r_fit[
            visible
        ],
        T_fit[
            visible
        ],
        "--",
        color="black",
        linewidth=1.5,
        zorder=4,
    )


format_axis(
    ax,
    r"$p=q$",
)


add_code_capacity_style_legend(
    ax
)


# =============================================================================
# Inset: lambda(d)
# =============================================================================

left = 0.64
bottom = 0.55
width = 0.24
height = 0.30


ax2 = fig.add_axes(
    [
        left,
        bottom,
        width,
        height,
    ]
)


# -----------------------------------------------------------------------------
# Theory
# -----------------------------------------------------------------------------

d_theory = np.linspace(
    1,
    max(
        DS
    ),
    500,
)


ax2.plot(
    d_theory,
    lambda_theory(
        d_theory
    ),
    color="purple",
    linewidth=1.5,
    zorder=2,
)


# -----------------------------------------------------------------------------
# Numerical fitted exponents
# -----------------------------------------------------------------------------

fit_ds = []
fit_lam = []
fit_err = []


for d in sorted(
    fits
):

    fit_ds.append(
        d
    )

    fit_lam.append(
        fits[
            d
        ][
            "lambda"
        ]
    )

    fit_err.append(
        fits[
            d
        ][
            "lambda_se"
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
    OUT_PHENO,
    bbox_inches="tight",
)

fig.savefig(
    OUT_PHENO_PNG,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


print(
    f"Saved: {OUT_PHENO}"
)

print(
    f"Saved: {OUT_PHENO_PNG}"
)

print(
    f"Saved: {OUT_FITS}"
)

print()
print(
    "=" * 88
)
