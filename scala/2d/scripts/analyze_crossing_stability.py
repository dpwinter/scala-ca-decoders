# scripts/analyze_crossing_stability.py

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT = Path("figs/scala2d_cc_crossings.csv")
OUTPUT = Path("figs/scala2d_cc_crossing_stability.csv")

D_EFF_CUTS = [
    0,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    120,
    140,
]


# ============================================================
# Load
# ============================================================

df = pd.read_csv(INPUT)

required = {
    "d1",
    "d2",
    "d_eff",
    "p_cross",
    "p_cross_err",
}

missing = required - set(df.columns)

if missing:
    raise RuntimeError(
        f"Missing columns: {sorted(missing)}"
    )

df = df[
    np.isfinite(df["d_eff"])
    & np.isfinite(df["p_cross"])
    & np.isfinite(df["p_cross_err"])
    & (df["p_cross_err"] > 0)
].copy()


# ============================================================
# Helpers
# ============================================================

def weighted_mean_and_error(x, sigma):
    w = 1.0 / sigma**2

    mean = np.sum(w * x) / np.sum(w)
    err = 1.0 / np.sqrt(np.sum(w))

    return mean, err


def chi2_red_about_mean(x, sigma, mean):
    chi2 = np.sum(
        ((x - mean) / sigma)**2
    )

    dof = len(x) - 1

    if dof <= 0:
        return np.nan

    return chi2 / dof


# ============================================================
# Stability scan
# ============================================================

rows = []

print()
print("=" * 110)
print("SCALA2D LARGE-d CROSSING STABILITY SCAN")
print("=" * 110)

print(
    f"{'d_eff min':>10} "
    f"{'N':>4} "
    f"{'weighted mean':>15} "
    f"{'stat err':>12} "
    f"{'unweighted':>15} "
    f"{'sample std':>12} "
    f"{'chi2_red':>12}"
)

print("-" * 110)


for dmin in D_EFF_CUTS:

    sub = df[
        df["d_eff"] >= dmin
    ].copy()

    if len(sub) < 2:
        continue

    x = sub["p_cross"].to_numpy(float)
    sigma = sub["p_cross_err"].to_numpy(float)

    weighted_mean, stat_err = weighted_mean_and_error(
        x,
        sigma,
    )

    unweighted_mean = np.mean(x)

    sample_std = np.std(
        x,
        ddof=1,
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

    rows.append({
        "d_eff_min": dmin,
        "N": len(sub),
        "weighted_mean": weighted_mean,
        "stat_err": stat_err,
        "unweighted_mean": unweighted_mean,
        "sample_std": sample_std,
        "chi2_red": chi2_red,
    })


# ============================================================
# Save
# ============================================================

out = pd.DataFrame(rows)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

out.to_csv(
    OUTPUT,
    index=False,
)


# ============================================================
# Additional diagnostics
# ============================================================

print()
print("=" * 110)
print("LARGEST-d CROSSINGS")
print("=" * 110)

largest = df.sort_values(
    "d_eff"
).tail(8)

print(
    largest[
        [
            "d1",
            "d2",
            "d_eff",
            "p_cross",
            "p_cross_err",
        ]
    ].to_string(
        index=False
    )
)

print()
print(f"Saved: {OUTPUT}")
