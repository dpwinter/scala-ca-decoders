"""
SCALA1D code-capacity performance.

Numerical SCALA1D data are compared with the exact global-majority-voting
logical error probability stored in the input CSV.

Run:
    python -m scripts.plot_code_capacity

Input:
    data/code_capacity/code_capacity.csv

Outputs:
    figs/scala1d_cc.pdf
    figs/scala1d_cc.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Paths
# =============================================================================

INPUT = Path(
    "data/code_capacity/code_capacity.csv"
)

OUTPUT_DIR = Path("figs")

OUTPUT_PDF = (
    OUTPUT_DIR
    / "scala1d_cc.pdf"
)

OUTPUT_PNG = (
    OUTPUT_DIR
    / "scala1d_cc.png"
)


# =============================================================================
# Configuration
# =============================================================================

PLOT_D = [
    3,
    9,
    21,
    51,
    111,
    201,
]

PC = 0.5

XMIN = 0.09
XMAX = 1.0

YMIN = 1e-6
YMAX = 1.0


# =============================================================================
# Plot style
# =============================================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
})


# =============================================================================
# Load data
# =============================================================================

df = pd.read_csv(
    INPUT
)

required = {
    "d",
    "p",
    "pL",
    "se",
    "pL_exact",
}

missing = (
    required
    - set(df.columns)
)

if missing:
    raise RuntimeError(
        f"Missing columns: {sorted(missing)}. "
        f"Available columns: {list(df.columns)}"
    )


df = df[
    df["d"].isin(PLOT_D)
].copy()

df = df[
    np.isfinite(df["p"])
    & np.isfinite(df["pL"])
    & np.isfinite(df["se"])
    & np.isfinite(df["pL_exact"])
    & (df["p"] > 0)
    & (df["pL"] > 0)
    & (df["pL_exact"] > 0)
].copy()


# =============================================================================
# Plot
# =============================================================================

fig = plt.figure(
    figsize=(5, 4)
)

ax = plt.gca()


# =============================================================================
# Numerical data
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
        yerr=sub["se"],
        linestyle="-",
        marker="o",
        markerfacecolor="none",
        capsize=2,
        label=str(d),
        zorder=3,
    )


# =============================================================================
# Exact global-majority-voting curves
# =============================================================================
#
# Draw these above the colored curves so that the agreement remains visible.
# They are omitted from the legend.
# =============================================================================

for d in PLOT_D:

    sub = (
        df[df["d"] == d]
        .sort_values("p")
        .copy()
    )

    if sub.empty:
        continue

    ax.plot(
        sub["p"],
        sub["pL_exact"],
        "--",
        color="black",
        linewidth=1.3,
        zorder=4,
    )


# =============================================================================
# Threshold
# =============================================================================

ax.axvline(
    PC,
    linestyle="--",
    color="red",
    linewidth=1.3,
    zorder=2,
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
    r"$p$"
)

ax.set_ylabel(
    r"$p_L$"
)

ax.grid()


# =============================================================================
# Legend
# =============================================================================

ax.legend(
    title=r"$d$",
    loc="lower right",
    ncol=1,
)

# =============================================================================
# Save
# =============================================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

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

print(
    f"Saved: {OUTPUT_PDF}"
)

print(
    f"Saved: {OUTPUT_PNG}"
)


if __name__ == "__main__":
    pass
