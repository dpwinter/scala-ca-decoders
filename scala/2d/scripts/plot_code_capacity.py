# scripts/plot_code_capacity.py

from functools import lru_cache
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

INPUT = Path("data/code_capacity/code_capacity.csv")

OUTPUT_DIR = Path("figs")
OUTPUT_PDF = OUTPUT_DIR / "scala2d_cc.pdf"
OUTPUT_PNG = OUTPUT_DIR / "scala2d_cc.png"
OUTPUT_FITS = OUTPUT_DIR / "scala2d_cc_tail_fits.csv"
OUTPUT_SCAN = OUTPUT_DIR / "scala2d_cc_fit_scan.csv"


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


# Distances for which the resolved low-p regime is used for an exponent fit.
FIT_D = [
    3,
    9,
    15,
    21,
    31,
]


# Final fit windows selected from the low-p stability analysis.
FIT_WINDOWS = {
    3:  (0.0010, 0.0150),
    9:  (0.0040, 0.0060),
    15: (0.0090, 0.0175),
    21: (0.0150, 0.0250),
    31: (0.0225, 0.0375),
}


# Largest p included in the stability scan.
SCAN_PMAX = {
    3:   0.0400,
    9:   0.0300,
    15:  0.0350,
    21:  0.0400,
    31:  0.0450,
    51:  0.0600,
    81:  0.0600,
    111: 0.0650,
}


MIN_FIT_POINTS = 3


# Main-panel plotting grid.
PLOT_P = np.array([
    0.0010,
    0.0020,
    0.0030,
    0.0040,
    0.0050,
    0.0060,
    0.0070,
    0.0080,
    0.0090,
    0.0100,
    0.0125,
    0.0150,
    0.0175,
    0.0200,
    0.0225,
    0.0250,
    0.0275,
    0.0300,
    0.0325,
    0.0350,
    0.0375,
    0.0400,
    0.0425,
    0.0450,
    0.0500,
    0.0550,
    0.0600,
    0.0650,
    0.0700,
    0.0750,
    0.0800,
    0.0850,
    0.0900,
    0.0950,
    0.1000,
    0.1250,
    0.2000,
    0.3000,
    0.4000,
    0.5000,
])


XMIN = 0.005
XMAX = 0.50

YMIN = 1e-6
YMAX = 1.0


# Large-d crossing estimate.
PC = 0.0740
PC_ERR = 0.0005


plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
})


# =============================================================================
# Exact discrete SCALA failure mechanism
# =============================================================================

@lru_cache(maxsize=None)
def factor_sum(w0):
    """
    Exact discrete quantity

        f(w0) = min_{k n_e = w0} (k + n_e),

    with k,n_e positive integers.
    """

    w0 = int(w0)

    return min(
        k + w0 // k
        for k in range(1, math.isqrt(w0) + 1)
        if w0 % k == 0
    )


@lru_cache(maxsize=None)
def grown_weight(w0):
    """
    Weight after the growth mechanism:

        w_max(w0) = 2 w0 - f(w0) + 1.
    """

    w0 = int(w0)

    return (
        2 * w0
        - factor_sum(w0)
        + 1
    )


@lru_cache(maxsize=None)
def expected_weight(d):
    """
    Exact discrete failure-weight prediction for odd d:

        w0(d) =
        min { w0 in Z_+ :
              2 w0 - f(w0) + 1 >= (d+1)/2 }.
    """

    d = int(d)

    if d % 2 == 0:
        raise ValueError(
            f"SCALA staircase is evaluated only for odd d; got d={d}."
        )

    target = (d + 1) // 2

    w0 = 1

    while grown_weight(w0) < target:
        w0 += 1

    return w0


def plateau_segments(d_min=1, d_max=119):
    """
    Group consecutive odd distances having the same exact w0(d).

    Returns:
        [(d_start, d_end, w0), ...]
    """

    ds = np.arange(
        d_min,
        d_max + 1,
        2,
        dtype=int,
    )

    ws = np.array(
        [expected_weight(int(d)) for d in ds],
        dtype=int,
    )

    segments = []

    start = ds[0]
    current_w = ws[0]

    for i in range(1, len(ds)):
        if ws[i] != current_w:
            segments.append(
                (
                    int(start),
                    int(ds[i - 1]),
                    int(current_w),
                )
            )

            start = ds[i]
            current_w = ws[i]

    segments.append(
        (
            int(start),
            int(ds[-1]),
            int(current_w),
        )
    )

    return segments


# =============================================================================
# Fit helpers
# =============================================================================

def weighted_linear_fit(X, y, sigma):

    weights = 1.0 / sigma**2

    XT_W = X.T * weights

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
        (residual / sigma)**2
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

    return beta, cov, chi2, chi2_red


def select_fit_data(
    sub,
    p_min,
    p_max,
):

    return (
        sub[
            (sub["p"] >= p_min)
            & (sub["p"] <= p_max)
            & (sub["pL_raw"] > 0.0)
            & (sub["se_raw"] > 0.0)
            & np.isfinite(sub["pL_raw"])
            & np.isfinite(sub["se_raw"])
        ]
        .sort_values("p")
        .copy()
    )


def fit_power_law(
    sub,
    p_min,
    p_max,
):

    fit_sub = select_fit_data(
        sub,
        p_min,
        p_max,
    )

    if len(fit_sub) < MIN_FIT_POINTS:
        return None

    p = fit_sub["p"].to_numpy(float)
    pL = fit_sub["pL_raw"].to_numpy(float)
    se = fit_sub["se_raw"].to_numpy(float)

    x = np.log(p)
    y = np.log(pL)

    sigma_log = (
        se / pL
    )

    X = np.column_stack([
        np.ones_like(x),
        x,
    ])

    beta, cov, chi2, chi2_red = (
        weighted_linear_fit(
            X,
            y,
            sigma_log,
        )
    )

    log_A, lam = beta

    return {
        "A": float(np.exp(log_A)),
        "lambda": float(lam),
        "lambda_err": float(
            np.sqrt(cov[1, 1])
        ),
        "chi2": float(chi2),
        "chi2_red": float(chi2_red),
        "N": len(fit_sub),
        "data": fit_sub,
    }


def fit_fixed_exponent(
    sub,
    p_min,
    p_max,
    lam,
):

    fit_sub = select_fit_data(
        sub,
        p_min,
        p_max,
    )

    if len(fit_sub) < MIN_FIT_POINTS:
        return None

    p = fit_sub["p"].to_numpy(float)
    pL = fit_sub["pL_raw"].to_numpy(float)
    se = fit_sub["se_raw"].to_numpy(float)

    x = np.log(p)
    y = np.log(pL)

    sigma_log = (
        se / pL
    )

    z = (
        y
        - lam * x
    )

    weights = (
        1.0
        / sigma_log**2
    )

    log_A = (
        np.sum(weights * z)
        / np.sum(weights)
    )

    y_fit = (
        log_A
        + lam * x
    )

    chi2 = np.sum(
        (
            (y - y_fit)
            / sigma_log
        )**2
    )

    dof = len(y) - 1

    return {
        "A": float(
            np.exp(log_A)
        ),
        "chi2": float(chi2),
        "chi2_red": (
            float(chi2 / dof)
            if dof > 0
            else np.nan
        ),
        "N": len(fit_sub),
    }


def power_law(
    p,
    A,
    lam,
):

    return (
        A
        * np.asarray(p, dtype=float)**lam
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
    "pL_raw",
    "se_raw",
}


missing = (
    required
    - set(df.columns)
)


if missing:
    raise RuntimeError(
        f"Missing columns: {sorted(missing)}"
    )


df = (
    df[
        df["d"].isin(PLOT_D)
        & (df["p"] > 0.0)
        & np.isfinite(df["p"])
        & np.isfinite(df["pL_raw"])
        & np.isfinite(df["se_raw"])
    ]
    .copy()
)


# =============================================================================
# Exact staircase sanity check
# =============================================================================

print()
print("=" * 92)
print("SCALA2D exact discrete failure weights")
print("=" * 92)


for d in PLOT_D:
    print(
        f"d={d:3d}  "
        f"w0={expected_weight(d):3d}"
    )


EXPECTED = {
    3: 2,
    9: 4,
    15: 6,
    21: 8,
    31: 12,
    51: 18,
    81: 25,
    111: 35,
}


for d, w in EXPECTED.items():
    assert expected_weight(d) == w


# =============================================================================
# Stability scan
# =============================================================================

scan_rows = []


print()
print("=" * 120)
print("SCALA2D raw-failure asymptotic fit stability scan")
print("=" * 120)


for d in PLOT_D:

    w_expected = (
        expected_weight(d)
    )

    sub = (
        df[
            (df["d"] == d)
            & (df["p"] <= SCAN_PMAX[d])
            & (df["pL_raw"] > 0.0)
            & (df["se_raw"] > 0.0)
        ]
        .sort_values("p")
        .copy()
    )


    print()
    print(
        f"d={d}, expected w0={w_expected}"
    )

    print(
        f"{'pmin':>8} "
        f"{'pmax':>8} "
        f"{'N':>4} "
        f"{'lambda':>11} "
        f"{'sigma':>10} "
        f"{'pull':>9} "
        f"{'chi2_free':>11} "
        f"{'chi2_fixed':>12}"
    )

    print("-" * 82)


    if len(sub) < MIN_FIT_POINTS:
        print("insufficient resolved points")
        continue


    p_values = (
        sub["p"]
        .to_numpy(float)
    )


    n_start = min(
        3,
        len(p_values)
        - MIN_FIT_POINTS
        + 1,
    )


    for i0 in range(n_start):

        for i1 in range(
            i0 + MIN_FIT_POINTS - 1,
            len(p_values),
        ):

            p_min = float(
                p_values[i0]
            )

            p_max = float(
                p_values[i1]
            )


            free = fit_power_law(
                sub,
                p_min,
                p_max,
            )

            fixed = fit_fixed_exponent(
                sub,
                p_min,
                p_max,
                w_expected,
            )


            if free is None or fixed is None:
                continue


            pull = (
                (
                    free["lambda"]
                    - w_expected
                )
                / free["lambda_err"]
            )


            print(
                f"{p_min:8.4f} "
                f"{p_max:8.4f} "
                f"{free['N']:4d} "
                f"{free['lambda']:11.5f} "
                f"{free['lambda_err']:10.5f} "
                f"{pull:9.2f} "
                f"{free['chi2_red']:11.3f} "
                f"{fixed['chi2_red']:12.3f}"
            )


            scan_rows.append({
                "d": d,
                "w_expected": w_expected,
                "p_min": p_min,
                "p_max": p_max,
                "N": free["N"],
                "lambda": free["lambda"],
                "lambda_err": free["lambda_err"],
                "pull_sigma": pull,
                "chi2_red_free": free["chi2_red"],
                "chi2_red_fixed": fixed["chi2_red"],
            })


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


pd.DataFrame(
    scan_rows
).to_csv(
    OUTPUT_SCAN,
    index=False,
)


# =============================================================================
# Final nominal fits
# =============================================================================

fits = {}
fit_rows = []


print()
print("=" * 110)
print("SCALA2D final raw-failure power-law fits")
print("=" * 110)


for d in PLOT_D:

    if d not in FIT_D:

        print(
            f"d={d:3d}: fit omitted "
            f"(expected w0={expected_weight(d)})"
        )

        continue


    p_min, p_max = (
        FIT_WINDOWS[d]
    )


    sub = (
        df[
            df["d"] == d
        ]
        .sort_values("p")
        .copy()
    )


    free = fit_power_law(
        sub,
        p_min,
        p_max,
    )

    fixed = fit_fixed_exponent(
        sub,
        p_min,
        p_max,
        expected_weight(d),
    )


    if free is None or fixed is None:

        print(
            f"d={d:3d}: insufficient points"
        )

        continue


    fits[d] = free


    pull = (
        (
            free["lambda"]
            - expected_weight(d)
        )
        / free["lambda_err"]
    )


    print(
        f"d={d:3d}  "
        f"window=[{p_min:.4f},{p_max:.4f}]  "
        f"lambda={free['lambda']:.6f} "
        f"+/- {free['lambda_err']:.6f}  "
        f"w0={expected_weight(d):2d}  "
        f"pull={pull:+.2f} sigma  "
        f"N={free['N']:2d}  "
        f"chi2_free={free['chi2_red']:.3f}  "
        f"chi2_fixed={fixed['chi2_red']:.3f}"
    )


    fit_rows.append({
        "d": d,
        "w_expected": expected_weight(d),
        "p_min": p_min,
        "p_max": p_max,
        "A": free["A"],
        "lambda": free["lambda"],
        "lambda_err": free["lambda_err"],
        "pull_sigma": pull,
        "N": free["N"],
        "chi2_red_free": free["chi2_red"],
        "chi2_red_fixed": fixed["chi2_red"],
    })


pd.DataFrame(
    fit_rows
).to_csv(
    OUTPUT_FITS,
    index=False,
)


# =============================================================================
# Plotting subset
# =============================================================================

def is_plot_p(p):

    return np.any(
        np.isclose(
            p,
            PLOT_P,
            rtol=0.0,
            atol=1e-10,
        )
    )


df_plot = (
    df[
        df["p"].apply(is_plot_p)
    ]
    .copy()
)


# Always retain the two lowest resolved points at every distance.
lowest = (
    df[
        (df["pL_raw"] > 0.0)
        & (df["se_raw"] > 0.0)
    ]
    .sort_values(
        ["d", "p"]
    )
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
    .sort_values(
        ["d", "p"]
    )
)


# =============================================================================
# Figure
# =============================================================================

fig = plt.figure(
    figsize=(5, 4)
)

ax = plt.gca()


# Threshold band.
ax.axvspan(
    PC - PC_ERR,
    PC + PC_ERR,
    color="red",
    alpha=0.08,
    zorder=0,
)

ax.axvline(
    PC,
    color="red",
    linestyle="--",
    linewidth=1.2,
    zorder=1,
)


# Monte Carlo curves.
for d in PLOT_D:

    sub = (
        df_plot[
            (df_plot["d"] == d)
            & (df_plot["pL_raw"] > 0.0)
        ]
        .sort_values("p")
    )

    if sub.empty:
        continue

    ax.errorbar(
        sub["p"],
        sub["pL_raw"],
        yerr=sub["se_raw"],
        linestyle="-",
        marker="o",
        markerfacecolor="none",
        capsize=2,
        label=str(d),
        zorder=2,
    )


# Tail fits: draw only over the fitted p interval.
for d, fit in fits.items():

    p_min = float(
        fit["data"]["p"].min()
    )

    p_max = float(
        fit["data"]["p"].max()
    )

    p_fit = np.logspace(
        np.log10(p_min),
        np.log10(p_max),
        400,
    )

    y_fit = power_law(
        p_fit,
        fit["A"],
        fit["lambda"],
    )

    visible = (
        (y_fit >= YMIN)
        & (y_fit <= YMAX)
    )

    ax.plot(
        p_fit[visible],
        y_fit[visible],
        "--",
        color="black",
        linewidth=1.5,
        zorder=4,
    )


ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlim(XMIN, XMAX)
ax.set_ylim(YMIN, YMAX)

ax.set_xlabel(r"$p$")
ax.set_ylabel(r"$p_L$")

ax.grid()


# Original legend layout.
ax.legend(
    title=r"$d$",
    bbox_to_anchor=(0, 1.02, 1, 0.2),
    loc="lower left",
    mode="expand",
    borderaxespad=0,
    ncol=4,
)


# =============================================================================
# Inset: exact discrete odd-d plateaus
# =============================================================================

ax2 = fig.add_axes([
    0.64,
    0.22,
    0.24,
    0.30,
])


# Draw only horizontal segments.
#
# Each allowed odd distance represents a width-2 interval, so a plateau
# d_start,...,d_end is shown from d_start-1 to d_end+1.
for d_start, d_end, w0 in plateau_segments(
    d_min=1,
    d_max=119,
):

    ax2.hlines(
        y=w0,
        xmin=max(0, d_start - 1),
        xmax=d_end + 1,
        color="purple",
        linewidth=1.5,
        zorder=2,
    )


# Numerical exponents.
for d in FIT_D:

    if d not in fits:
        continue

    ax2.errorbar(
        d,
        fits[d]["lambda"],
        yerr=fits[d]["lambda_err"],
        linestyle="none",
        marker="o",
        markersize=5,
        markerfacecolor="none",
        color="black",
        capsize=2,
        zorder=3,
    )


ax2.set_xlim(0, 120)
ax2.set_ylim(0, 38)

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


print()
print("=" * 110)
print(f"Saved: {OUTPUT_PDF}")
print(f"Saved: {OUTPUT_PNG}")
print(f"Saved: {OUTPUT_FITS}")
print(f"Saved: {OUTPUT_SCAN}")
print("=" * 110)
