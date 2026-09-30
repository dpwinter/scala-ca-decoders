# scripts/plot_pheno_signal_noise.py

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

INPUT_CLEAN = Path(
    "data/pheno/pheno_pL_fixed.csv"
)

INPUT_SIGNAL = Path(
    "data/pheno/pheno_pL_signal_noise.csv"
)

OUTPUT_DIR = Path("figs")

OUTPUT_PDF = (
    OUTPUT_DIR
    / "scala1d_pheno_signal_noise.pdf"
)

OUTPUT_PNG = (
    OUTPUT_DIR
    / "scala1d_pheno_signal_noise.png"
)


PLOT_D = [
    3,
    9,
    31,
    81,
]


XMIN = 1e-2
XMAX = 1e-1

YMIN = 1e-8
YMAX = 1e-1


# Opacity of p_sig = 0 reference curves.
CLEAN_ALPHA = 0.24


plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 14,
    "legend.fontsize": 10,
})


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


def load_stationary(
    path,
    require_signal_noise=False,
):

    df = pd.read_csv(path)

    required = {
        "d",
        "p",
        "q",
        "stationary",
        "pL",
        "pL_se",
    }

    if require_signal_noise:
        required.add("p_sig")

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"{path}: missing columns "
            f"{sorted(missing)}"
        )


    stationary = as_bool(
        df["stationary"]
    )


    mask = (
        stationary
        & df["d"].isin(PLOT_D)
        & np.isclose(
            df["p"],
            df["q"],
            atol=1e-12,
            rtol=0.0,
        )
        & (df["p"] >= XMIN)
        & (df["p"] <= XMAX)
        & np.isfinite(df["p"])
        & np.isfinite(df["pL"])
        & np.isfinite(df["pL_se"])
        & (df["pL"] > 0.0)
        & (df["pL_se"] > 0.0)
    )


    if require_signal_noise:

        # Signal-noise data must satisfy
        #
        #     p = q = p_sig.
        #
        mask &= np.isclose(
            df["p"],
            df["p_sig"],
            atol=1e-12,
            rtol=0.0,
        )


    return (
        df[mask]
        .copy()
        .sort_values(
            ["d", "p"]
        )
    )


# =============================================================================
# Load
# =============================================================================

clean = load_stationary(
    INPUT_CLEAN,
    require_signal_noise=False,
)

signal = load_stationary(
    INPUT_SIGNAL,
    require_signal_noise=True,
)


print()
print("=" * 100)
print("SCALA1D phenomenological signal-noise comparison")
print("=" * 100)


for d in PLOT_D:

    clean_d = (
        clean[
            clean["d"] == d
        ]
        .sort_values("p")
    )

    signal_d = (
        signal[
            signal["d"] == d
        ]
        .sort_values("p")
    )

    print(
        f"d={d:3d}: "
        f"clean={len(clean_d):2d} resolved, "
        f"signal={len(signal_d):2d} resolved"
    )

    if not signal_d.empty:

        print(
            "       signal p: "
            + " ".join(
                f"{p:.4f}"
                for p in signal_d["p"]
            )
        )


# =============================================================================
# Figure
# =============================================================================

fig, ax = plt.subplots(
    figsize=(5.0, 4.0)
)


# =============================================================================
# Colors
# =============================================================================

colors = (
    plt.rcParams[
        "axes.prop_cycle"
    ]
    .by_key()["color"]
)


color_for_d = {
    d: colors[
        i % len(colors)
    ]
    for i, d
    in enumerate(PLOT_D)
}


# =============================================================================
# Clean-signal reference
#
# p = q
# p_sig = 0
# =============================================================================

for d in PLOT_D:

    sub = (
        clean[
            clean["d"] == d
        ]
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
        linewidth=1.5,
        marker="o",
        markersize=5,
        markerfacecolor="none",
        markeredgewidth=1.0,
        color=color_for_d[d],
        alpha=CLEAN_ALPHA,
        capsize=2,
        zorder=1,
    )


# =============================================================================
# Signal-noise data
#
# p = q = p_sig
# =============================================================================

for d in PLOT_D:

    sub = (
        signal[
            signal["d"] == d
        ]
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
        linewidth=1.7,
        marker="o",
        markersize=5,
        markerfacecolor="none",
        markeredgewidth=1.1,
        color=color_for_d[d],
        alpha=1.0,
        capsize=2,
        zorder=3,
    )


# =============================================================================
# Axes
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
    r"$p=q=p_{\rm sig}$"
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
    )
)

ax.xaxis.set_minor_formatter(
    NullFormatter()
)


ax.yaxis.set_major_locator(
    LogLocator(
        base=10.0,
        subs=(1.0,),
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
    )
)

ax.yaxis.set_minor_formatter(
    NullFormatter()
)


ax.grid(
    True,
    which="major",
)


# =============================================================================
# Distance legend
#
# Lower-right, one column.
# =============================================================================

distance_handles = []

for d in PLOT_D:

    distance_handles.append(
        Line2D(
            [0],
            [0],
            color=color_for_d[d],
            linestyle="-",
            marker="o",
            markerfacecolor="none",
            markeredgewidth=1.0,
            markersize=5,
            linewidth=1.5,
            label=str(d),
        )
    )


ax.legend(
    handles=distance_handles,
    title=r"$d$",
    loc="lower right",
    ncol=1,
    frameon=True,
    borderpad=0.5,
    handlelength=1.8,
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
print("=" * 100)
print(f"Saved: {OUTPUT_PDF}")
print(f"Saved: {OUTPUT_PNG}")
print("=" * 100)
