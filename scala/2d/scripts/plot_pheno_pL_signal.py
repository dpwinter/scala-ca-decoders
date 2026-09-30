# scripts/plot_pheno_pL_signal.py

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import (
    LogFormatterMathtext,
    LogLocator,
    NullFormatter,
)
import numpy as np
import pandas as pd


# =============================================================================
# Input / output
# =============================================================================

BASELINE_CSV = Path(
    "data/pheno/scala2d_pheno_pL_staircase.csv"
)

SIGNAL_CSV = Path(
    "data/pheno/scala2d_pheno_pL_staircase_signal.csv"
)

OUTPUT_DIR = Path(
    "figs"
)

OUTPUT_PDF = (
    OUTPUT_DIR
    / "scala2d_pheno_pL_signal_noise.pdf"
)

OUTPUT_PNG = (
    OUTPUT_DIR
    / "scala2d_pheno_pL_signal_noise.png"
)


# =============================================================================
# Plot configuration
# =============================================================================

PLOT_D = [
    5,
    9,
    15,
    21,
    31,
]

COLORS = {
    5: "tab:blue",
    9: "tab:orange",
    15: "tab:green",
    21: "tab:red",
    31: "tab:purple",
}


# =============================================================================
# Reduced display set
#
# This affects only the number of markers drawn, not the stored data.
# =============================================================================

MAX_PLOT_POINTS = 10


# =============================================================================
# Axes
# =============================================================================

XMIN = 1.0e-3
XMAX = 2.2e-2

YMIN = 1.0e-8
YMAX = 1.0


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 14,
        "legend.fontsize": 11,
    }
)


# =============================================================================
# Helpers
# =============================================================================

def as_bool(series):

    if series.dtype == bool:
        return series

    return (
        series
        .astype(str)
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


def reduced_plot_subset(
    sub,
    max_points=MAX_PLOT_POINTS,
):
    """
    Select a reduced set of actual Monte Carlo points.

    Points are chosen approximately uniformly in log(p).
    No interpolation is performed.
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


def get_resolved_curve(
    df,
    d,
):

    sub = df[
        (
            df["d"] == d
        )
        & (
            df["stationary"]
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
        & (
            df["p"] > 0
        )
        & (
            df["pL"] > 0
        )
    ].copy()

    return (
        sub
        .sort_values("p")
    )


# =============================================================================
# Load baseline
# =============================================================================

baseline = pd.read_csv(
    BASELINE_CSV
)

required = {
    "d",
    "p",
    "q",
    "p_sig",
    "stationary",
    "pL",
    "pL_se",
}

missing = (
    required
    - set(
        baseline.columns
    )
)

if missing:

    raise RuntimeError(
        f"Missing columns in {BASELINE_CSV}: "
        f"{sorted(missing)}"
    )


baseline["stationary"] = as_bool(
    baseline["stationary"]
)


baseline = baseline[
    baseline["d"].isin(
        PLOT_D
    )
    & np.isclose(
        baseline["p"],
        baseline["q"],
        atol=1e-12,
        rtol=0.0,
    )
    & np.isclose(
        baseline["p_sig"],
        0.0,
        atol=1e-15,
        rtol=0.0,
    )
].copy()


# =============================================================================
# Load signal-noise data
# =============================================================================

signal = pd.read_csv(
    SIGNAL_CSV
)

missing = (
    required
    - set(
        signal.columns
    )
)

if missing:

    raise RuntimeError(
        f"Missing columns in {SIGNAL_CSV}: "
        f"{sorted(missing)}"
    )


signal["stationary"] = as_bool(
    signal["stationary"]
)


signal = signal[
    signal["d"].isin(
        PLOT_D
    )
    & np.isclose(
        signal["p"],
        signal["q"],
        atol=1e-12,
        rtol=0.0,
    )
    & np.isclose(
        signal["p"],
        signal["p_sig"],
        atol=1e-12,
        rtol=0.0,
    )
].copy()


# =============================================================================
# Diagnostics
# =============================================================================

print()
print("=" * 88)

print(
    "SCALA2D PHENOMENOLOGICAL SIGNAL-NOISE COMPARISON"
)

print("=" * 88)


for d in PLOT_D:

    base_sub = get_resolved_curve(
        baseline,
        d,
    )

    sig_sub = get_resolved_curve(
        signal,
        d,
    )

    print(
        f"d={d:2d}: "
        f"baseline resolved={len(base_sub):2d}, "
        f"signal resolved={len(sig_sub):2d}"
    )


print("=" * 88)


# =============================================================================
# Figure
# =============================================================================

fig, ax = plt.subplots(
    figsize=(
        5,
        4,
    )
)


# =============================================================================
# Curves
# =============================================================================

for d in PLOT_D:

    color = COLORS[
        d
    ]


    # -------------------------------------------------------------------------
    # Baseline:
    #
    # p = q
    # p_sig = 0
    #
    # faint
    # -------------------------------------------------------------------------

    sub = get_resolved_curve(
        baseline,
        d,
    )

    if not sub.empty:

        sub = reduced_plot_subset(
            sub,
            MAX_PLOT_POINTS,
        )

        ax.errorbar(
            sub["p"],
            sub["pL"],
            yerr=sub["pL_se"],
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
            alpha=0.22,
            zorder=2,
        )


    # -------------------------------------------------------------------------
    # Signal noise:
    #
    # p = q = p_sig
    #
    # solid
    # -------------------------------------------------------------------------

    sub = get_resolved_curve(
        signal,
        d,
    )

    if not sub.empty:

        sub = reduced_plot_subset(
            sub,
            MAX_PLOT_POINTS,
        )

        ax.errorbar(
            sub["p"],
            sub["pL"],
            yerr=sub["pL_se"],
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
            alpha=1.0,
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
# Log ticks
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
#
# Major only, same as the updated phenomenological baseline plot.
# =============================================================================

ax.grid(
    True,
    which="major",
    alpha=0.6,
)


# =============================================================================
# Legend inside plot
#
# Distance is encoded by color.
# The faint/solid distinction is explained in the caption, as in the
# Harrington signal-noise figure.
# =============================================================================

handles = []

for d in PLOT_D:

    h, = ax.plot(
        [],
        [],
        color=COLORS[d],
        linestyle="-",
        linewidth=1.5,
        marker="o",
        markersize=5,
        markerfacecolor="none",
        label=str(d),
    )

    handles.append(
        h
    )


ax.legend(
    handles=handles,
    title=r"$d$",
    loc="lower right",
    ncol=1,
    frameon=True,
)


# =============================================================================
# Save
# =============================================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


fig.tight_layout()


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

print(
    f"Saved: {OUTPUT_PDF}"
)

print(
    f"Saved: {OUTPUT_PNG}"
)

print("=" * 88)
