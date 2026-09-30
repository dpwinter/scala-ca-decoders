from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT = Path("data/code_capacity/code_capacity_fss.csv")

OUTPUT_DIR = Path("figs")
OUTPUT_PDF = OUTPUT_DIR / "scala2d_cc_crossings.pdf"
OUTPUT_PNG = OUTPUT_DIR / "scala2d_cc_crossings.png"
OUTPUT_CSV = OUTPUT_DIR / "scala2d_cc_crossings.csv"
OUTPUT_STABILITY = OUTPUT_DIR / "scala2d_cc_crossing_stability.csv"

P_MIN = 0.071
P_MAX = 0.078

N_BOOT = 3000
SEED = 12345
TARGET_P = 0.0745


# Representative subset shown in the paper figure.
# All crossings are still computed and saved.
PLOT_PAIRS = [
    (25, 27),
    (31, 41),
    (49, 51),
    (65, 67),
    (81, 101),
    (101, 121),
    (141, 161),
    (161, 181),
]


# Large-d threshold analysis
D_EFF_THRESHOLD = 80.0

STABILITY_CUTS = [
    80.0,
    90.0,
    100.0,
    120.0,
    140.0,
]


# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 14,
})


# ============================================================
# Weighted isotonic regression
# ============================================================

def isotonic_increasing(y, sigma):

    y = np.asarray(y, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    weights = 1.0 / sigma**2
    blocks = []

    for i in range(len(y)):

        blocks.append({
            "start": i,
            "end": i,
            "weight": weights[i],
            "mean": y[i],
        })

        while (
            len(blocks) >= 2
            and blocks[-2]["mean"] > blocks[-1]["mean"]
        ):

            b2 = blocks.pop()
            b1 = blocks.pop()

            w = (
                b1["weight"]
                + b2["weight"]
            )

            mean = (
                b1["weight"] * b1["mean"]
                + b2["weight"] * b2["mean"]
            ) / w

            blocks.append({
                "start": b1["start"],
                "end": b2["end"],
                "weight": w,
                "mean": mean,
            })

    out = np.empty_like(y)

    for b in blocks:
        out[
            b["start"]:
            b["end"] + 1
        ] = b["mean"]

    return out


# ============================================================
# Crossing finder
# ============================================================

def find_crossing(p, y1, y2):

    diff = y1 - y2
    crossings = []

    for i in range(len(p) - 1):

        a = diff[i]
        b = diff[i + 1]

        if a == 0:
            crossings.append(
                p[i]
            )
            continue

        if a * b > 0:
            continue

        p1 = p[i]
        p2 = p[i + 1]

        if np.isclose(a, b):

            px = (
                0.5
                * (p1 + p2)
            )

        else:

            px = (
                p1
                - a
                * (p2 - p1)
                / (b - a)
            )

        crossings.append(
            px
        )

    if not crossings:
        return np.nan

    return min(
        crossings,
        key=lambda x: abs(
            x - TARGET_P
        ),
    )


# ============================================================
# Pair preparation
# ============================================================

def prepare_pair(s1, s2):

    m = pd.merge(
        s1[
            ["p", "pL", "se"]
        ],
        s2[
            ["p", "pL", "se"]
        ],
        on="p",
        suffixes=("1", "2"),
    ).sort_values("p")

    m = m[
        (m["p"] >= P_MIN)
        & (m["p"] <= P_MAX)
        & np.isfinite(m["pL1"])
        & np.isfinite(m["pL2"])
        & np.isfinite(m["se1"])
        & np.isfinite(m["se2"])
        & (m["se1"] > 0)
        & (m["se2"] > 0)
    ].copy()

    return m


# ============================================================
# Central crossing
# ============================================================

def crossing_for_pair(s1, s2):

    m = prepare_pair(
        s1,
        s2,
    )

    if len(m) < 3:
        return np.nan

    p = m["p"].to_numpy(float)

    y1 = isotonic_increasing(
        m["pL1"].to_numpy(float),
        m["se1"].to_numpy(float),
    )

    y2 = isotonic_increasing(
        m["pL2"].to_numpy(float),
        m["se2"].to_numpy(float),
    )

    return find_crossing(
        p,
        y1,
        y2,
    )


# ============================================================
# Bootstrap crossing uncertainty
# ============================================================

def bootstrap_crossing(s1, s2, rng):

    m = prepare_pair(
        s1,
        s2,
    )

    if len(m) < 3:
        return np.nan, 0

    p = m["p"].to_numpy(float)

    mu1 = m["pL1"].to_numpy(float)
    mu2 = m["pL2"].to_numpy(float)

    se1 = m["se1"].to_numpy(float)
    se2 = m["se2"].to_numpy(float)

    samples = []

    for _ in range(N_BOOT):

        y1 = rng.normal(
            mu1,
            se1,
        )

        y2 = rng.normal(
            mu2,
            se2,
        )

        y1 = np.clip(
            y1,
            0.0,
            1.0,
        )

        y2 = np.clip(
            y2,
            0.0,
            1.0,
        )

        y1 = isotonic_increasing(
            y1,
            se1,
        )

        y2 = isotonic_increasing(
            y2,
            se2,
        )

        px = find_crossing(
            p,
            y1,
            y2,
        )

        if np.isfinite(px):
            samples.append(
                px
            )

    if len(samples) < 20:
        return np.nan, len(samples)

    samples = np.asarray(
        samples
    )

    return (
        float(
            np.std(
                samples,
                ddof=1,
            )
        ),
        len(samples),
    )


# ============================================================
# Mean / stability helpers
# ============================================================

def weighted_mean_and_error(x, sigma):

    w = 1.0 / sigma**2

    mean = (
        np.sum(w * x)
        / np.sum(w)
    )

    err = (
        1.0
        / np.sqrt(
            np.sum(w)
        )
    )

    return mean, err


def chi2_red_about_mean(x, sigma, mean):

    chi2 = np.sum(
        (
            (x - mean)
            / sigma
        )**2
    )

    dof = (
        len(x) - 1
    )

    if dof <= 0:
        return np.nan

    return (
        chi2 / dof
    )


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(
    INPUT
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
    & (df["p"] >= P_MIN)
    & (df["p"] <= P_MAX)
].copy()

df["d"] = (
    df["d"]
    .astype(int)
)

distances = sorted(
    df["d"].unique()
)


# ============================================================
# Compute all adjacent-pair crossings
# ============================================================

rng = np.random.default_rng(
    SEED
)

rows = []


for d1, d2 in zip(
    distances[:-1],
    distances[1:],
):

    s1 = df[
        df["d"] == d1
    ]

    s2 = df[
        df["d"] == d2
    ]

    p_cross = crossing_for_pair(
        s1,
        s2,
    )

    if not np.isfinite(
        p_cross
    ):
        continue

    p_err, n_success = bootstrap_crossing(
        s1,
        s2,
        rng,
    )

    if not np.isfinite(
        p_err
    ):
        continue

    rows.append({
        "d1": d1,
        "d2": d2,

        "d_eff": np.sqrt(
            d1 * d2
        ),

        "p_cross": p_cross,
        "p_cross_err": p_err,

        "bootstrap_successes": n_success,
    })


cross = pd.DataFrame(
    rows
)

if cross.empty:
    raise RuntimeError(
        "No pairwise crossings found."
    )


# ============================================================
# Large-d stability scan
# ============================================================

stability_rows = []

print()
print("=" * 105)
print("SCALA2D LARGE-d CROSSING STABILITY SCAN")
print("=" * 105)

print(
    f"{'d_eff min':>10} "
    f"{'N':>4} "
    f"{'weighted mean':>15} "
    f"{'stat err':>12} "
    f"{'unweighted':>15} "
    f"{'sample std':>12} "
    f"{'chi2_red':>12}"
)

print("-" * 105)


for dmin in STABILITY_CUTS:

    sub = cross[
        cross["d_eff"] >= dmin
    ].copy()

    if len(sub) < 2:
        continue

    x = sub[
        "p_cross"
    ].to_numpy(float)

    sigma = sub[
        "p_cross_err"
    ].to_numpy(float)

    weighted_mean, stat_err = (
        weighted_mean_and_error(
            x,
            sigma,
        )
    )

    unweighted_mean = float(
        np.mean(x)
    )

    sample_std = float(
        np.std(
            x,
            ddof=1,
        )
    )

    chi2_red = chi2_red_about_mean(
        x,
        sigma,
        weighted_mean,
    )

    print(
        f"{dmin:10.1f} "
        f"{len(sub):4d} "
        f"{weighted_mean:15.7f} "
        f"{stat_err:12.7f} "
        f"{unweighted_mean:15.7f} "
        f"{sample_std:12.7f} "
        f"{chi2_red:12.3f}"
    )

    stability_rows.append({
        "d_eff_min": dmin,
        "N": len(sub),

        "weighted_mean": weighted_mean,
        "stat_err": stat_err,

        "unweighted_mean": unweighted_mean,
        "sample_std": sample_std,

        "chi2_red": chi2_red,
    })


stability = pd.DataFrame(
    stability_rows
)


# ============================================================
# Final threshold estimate
# ============================================================
#
# Central value:
#   inverse-variance weighted mean for d_eff >= 80.
#
# Systematic uncertainty:
#   max(
#       sample standard deviation for d_eff >= 80,
#       maximum shift of weighted mean under the tested
#       larger-d cuts
#   )
#
# ============================================================

large = cross[
    cross["d_eff"]
    >= D_EFF_THRESHOLD
].copy()

if len(large) < 3:
    raise RuntimeError(
        "Too few large-d crossings "
        "for threshold estimate."
    )


x_large = large[
    "p_cross"
].to_numpy(float)

s_large = large[
    "p_cross_err"
].to_numpy(float)


PC, PC_STAT = weighted_mean_and_error(
    x_large,
    s_large,
)


PC_SCATTER = float(
    np.std(
        x_large,
        ddof=1,
    )
)


stable_means = (
    stability[
        stability["d_eff_min"]
        >= D_EFF_THRESHOLD
    ]["weighted_mean"]
    .to_numpy(float)
)


PC_CUT_SHIFT = float(
    np.max(
        np.abs(
            stable_means - PC
        )
    )
)


PC_ERR = max(
    PC_SCATTER,
    PC_CUT_SHIFT,
)


# Round to paper-level precision
PC_PLOT = round(
    PC,
    4,
)

PC_ERR_PLOT = round(
    PC_ERR,
    4,
)


print()
print("=" * 105)
print("FINAL LARGE-d CROSSING ESTIMATE")
print("=" * 105)

print(
    f"d_eff >= {D_EFF_THRESHOLD:.0f}"
)

print(
    f"weighted mean       = "
    f"{PC:.7f}"
)

print(
    f"statistical error   = "
    f"{PC_STAT:.7f}"
)

print(
    f"sample std          = "
    f"{PC_SCATTER:.7f}"
)

print(
    f"max cut shift       = "
    f"{PC_CUT_SHIFT:.7f}"
)

print(
    f"assigned uncertainty= "
    f"{PC_ERR:.7f}"
)

print()

print(
    f"paper estimate      = "
    f"{PC_PLOT:.4f} +/- "
    f"{PC_ERR_PLOT:.4f}"
)

print("=" * 105)


# ============================================================
# Representative subset for plotting
# ============================================================

plot_mask = np.zeros(
    len(cross),
    dtype=bool,
)

for d1, d2 in PLOT_PAIRS:

    plot_mask |= (
        (cross["d1"] == d1)
        & (cross["d2"] == d2)
    )


cross_plot = cross[
    plot_mask
].copy()

if cross_plot.empty:
    raise RuntimeError(
        "None of PLOT_PAIRS were found."
    )


# ============================================================
# Save numerical results
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

cross.to_csv(
    OUTPUT_CSV,
    index=False,
)

stability.to_csv(
    OUTPUT_STABILITY,
    index=False,
)


# ============================================================
# Crossing figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(5, 4)
)


# Final threshold uncertainty band
ax.axhspan(
    PC_PLOT - PC_ERR_PLOT,
    PC_PLOT + PC_ERR_PLOT,

    color="red",
    alpha=0.08,

    zorder=0,
)


# Final threshold central value
ax.axhline(
    PC_PLOT,

    color="red",
    linestyle="--",
    linewidth=1.2,

    zorder=1,
)


# Representative crossing estimates
ax.errorbar(
    cross_plot["d_eff"],
    cross_plot["p_cross"],

    yerr=cross_plot["p_cross_err"],

    linestyle="none",

    marker="o",
    markerfacecolor="none",

    capsize=3,

    zorder=2,
)


ax.set_xlabel(
    r"$d_{\rm eff}=\sqrt{d_1d_2}$"
)

ax.set_ylabel(
    r"$p_\times$"
)

ax.grid()


# ============================================================
# Save figure
# ============================================================

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
print(f"Saved: {OUTPUT_PDF}")
print(f"Saved: {OUTPUT_PNG}")
print(f"Saved: {OUTPUT_CSV}")
print(f"Saved: {OUTPUT_STABILITY}")
