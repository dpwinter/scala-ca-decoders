# scripts/analyze_code_capacity_fit_stability.py

from functools import lru_cache
from pathlib import Path
import math

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

INPUT = Path("data/code_capacity/code_capacity.csv")

OUTPUT_DIR = Path("figs")
OUTPUT_SCAN = OUTPUT_DIR / "scala2d_cc_fit_stability.csv"
OUTPUT_SELECTED = OUTPUT_DIR / "scala2d_cc_selected_fits.csv"


D_VALUES = [
    3,
    9,
    15,
    21,
    31,
    51,
    81,
    111,
]


# Maximum p included in the window scan.
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


# Final windows selected from the stability analysis.
SELECTED_WINDOWS = {
    3:  (0.0010, 0.0150),
    9:  (0.0040, 0.0060),
    15: (0.0090, 0.0175),
    21: (0.0150, 0.0250),
    31: (0.0225, 0.0375),
}


MIN_FIT_POINTS = 3

# Number of lowest resolved p values allowed as possible p_min values.
N_START_POINTS = 3


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
    Weight reached by the analytical failure mechanism:

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
        min {w0 in Z_+ :
             2 w0 - f(w0) + 1 >= (d+1)/2 }.
    """

    d = int(d)

    if d % 2 == 0:
        raise ValueError(
            f"Expected odd code distance, got d={d}."
        )

    target = (d + 1) // 2

    w0 = 1

    while grown_weight(w0) < target:
        w0 += 1

    return w0


# =============================================================================
# Fit routines
# =============================================================================

def select_fit_data(
    sub,
    p_min,
    p_max,
):
    """
    Select resolved raw-failure points in a candidate p window.
    """

    return (
        sub[
            (sub["p"] >= p_min)
            & (sub["p"] <= p_max)
            & np.isfinite(sub["pL_raw"])
            & np.isfinite(sub["se_raw"])
            & (sub["pL_raw"] > 0.0)
            & (sub["se_raw"] > 0.0)
        ]
        .sort_values("p")
        .copy()
    )


def weighted_linear_fit(
    X,
    y,
    sigma,
):
    """
    Weighted least squares with known pointwise standard deviations.
    """

    weights = 1.0 / sigma**2

    XT_W = (
        X.T
        * weights
    )

    normal = (
        XT_W @ X
    )

    cov = np.linalg.pinv(
        normal
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

    chi2 = float(
        np.sum(
            (residual / sigma)**2
        )
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


def fit_free_exponent(
    sub,
    p_min,
    p_max,
):
    """
    Fit

        p_L = A p^lambda

    by weighted least squares in log space.
    """

    fit_sub = select_fit_data(
        sub,
        p_min,
        p_max,
    )

    if len(fit_sub) < MIN_FIT_POINTS:
        return None

    p = (
        fit_sub["p"]
        .to_numpy(float)
    )

    pL = (
        fit_sub["pL_raw"]
        .to_numpy(float)
    )

    se = (
        fit_sub["se_raw"]
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


    log_A = float(
        beta[0]
    )

    lam = float(
        beta[1]
    )

    lam_err = float(
        np.sqrt(
            cov[1, 1]
        )
    )


    return {
        "A": float(
            np.exp(log_A)
        ),
        "lambda": lam,
        "lambda_err": lam_err,
        "chi2": chi2,
        "chi2_red": chi2_red,
        "N": int(
            len(fit_sub)
        ),
        "data": fit_sub,
    }


def fit_fixed_exponent(
    sub,
    p_min,
    p_max,
    lam,
):
    """
    Fit only A with lambda fixed:

        p_L = A p^lambda.
    """

    fit_sub = select_fit_data(
        sub,
        p_min,
        p_max,
    )

    if len(fit_sub) < MIN_FIT_POINTS:
        return None


    p = (
        fit_sub["p"]
        .to_numpy(float)
    )

    pL = (
        fit_sub["pL_raw"]
        .to_numpy(float)
    )

    se = (
        fit_sub["se_raw"]
        .to_numpy(float)
    )


    x = np.log(p)
    y = np.log(pL)

    sigma_log = (
        se / pL
    )


    z = (
        y
        - float(lam) * x
    )

    weights = (
        1.0
        / sigma_log**2
    )


    log_A = float(
        np.sum(
            weights * z
        )
        / np.sum(weights)
    )


    y_fit = (
        log_A
        + float(lam) * x
    )


    chi2 = float(
        np.sum(
            (
                (y - y_fit)
                / sigma_log
            )**2
        )
    )


    dof = (
        len(y) - 1
    )

    chi2_red = (
        chi2 / dof
        if dof > 0
        else np.nan
    )


    return {
        "A": float(
            np.exp(log_A)
        ),
        "chi2": chi2,
        "chi2_red": chi2_red,
        "N": int(
            len(fit_sub)
        ),
    }


# =============================================================================
# Load and validate input
# =============================================================================

df = pd.read_csv(
    INPUT
)


required_columns = {
    "d",
    "p",
    "pL_raw",
    "se_raw",
}


missing = (
    required_columns
    - set(df.columns)
)


if missing:
    raise RuntimeError(
        f"Missing required columns: "
        f"{sorted(missing)}"
    )


df = (
    df[
        df["d"].isin(D_VALUES)
        & np.isfinite(df["p"])
        & np.isfinite(df["pL_raw"])
        & np.isfinite(df["se_raw"])
        & (df["p"] > 0.0)
    ]
    .copy()
)


# =============================================================================
# Verify analytical weights
# =============================================================================

print()
print("=" * 104)
print("SCALA2D EXACT DISCRETE FAILURE-WEIGHT PREDICTION")
print("=" * 104)


for d in D_VALUES:

    print(
        f"d={d:3d}  "
        f"w0(d)={expected_weight(d):3d}"
    )


# =============================================================================
# Full stability scan
# =============================================================================

scan_rows = []


print()
print("=" * 120)
print("SCALA2D CODE-CAPACITY LOW-p FIT STABILITY SCAN")
print("=" * 120)


for d in D_VALUES:

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
        f"d = {d}, "
        f"w0(d) = {w_expected}"
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

    print(
        "-" * 82
    )


    if len(sub) < MIN_FIT_POINTS:

        print(
            "insufficient resolved points"
        )

        continue


    p_values = (
        sub["p"]
        .to_numpy(float)
    )


    n_start = min(
        N_START_POINTS,
        len(p_values)
        - MIN_FIT_POINTS
        + 1,
    )


    for i0 in range(
        n_start
    ):

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


            free = fit_free_exponent(
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


            if (
                free is None
                or fixed is None
            ):
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
                "A_free": free["A"],
                "lambda": free["lambda"],
                "lambda_err": free["lambda_err"],
                "pull_sigma": pull,
                "chi2_free": free["chi2"],
                "chi2_red_free": free["chi2_red"],
                "A_fixed": fixed["A"],
                "chi2_fixed": fixed["chi2"],
                "chi2_red_fixed": fixed["chi2_red"],
            })


# =============================================================================
# Selected final fits
# =============================================================================

selected_rows = []


print()
print("=" * 120)
print("SELECTED SCALA2D CODE-CAPACITY ASYMPTOTIC FITS")
print("=" * 120)


print(
    f"{'d':>4} "
    f"{'w0':>4} "
    f"{'window':>19} "
    f"{'N':>4} "
    f"{'lambda':>11} "
    f"{'sigma':>10} "
    f"{'pull':>9} "
    f"{'chi2_free':>11} "
    f"{'chi2_fixed':>12}"
)

print(
    "-" * 104
)


for d in D_VALUES:

    if d not in SELECTED_WINDOWS:

        print(
            f"{d:4d} "
            f"{expected_weight(d):4d} "
            f"{'omitted':>19}"
        )

        continue


    p_min, p_max = (
        SELECTED_WINDOWS[d]
    )


    sub = (
        df[
            df["d"] == d
        ]
        .sort_values("p")
        .copy()
    )


    free = fit_free_exponent(
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


    if (
        free is None
        or fixed is None
    ):

        raise RuntimeError(
            f"Selected fit for d={d} "
            f"contains too few resolved points."
        )


    w_expected = (
        expected_weight(d)
    )


    pull = (
        (
            free["lambda"]
            - w_expected
        )
        / free["lambda_err"]
    )


    window_text = (
        f"[{p_min:.4f},{p_max:.4f}]"
    )


    print(
        f"{d:4d} "
        f"{w_expected:4d} "
        f"{window_text:>19} "
        f"{free['N']:4d} "
        f"{free['lambda']:11.5f} "
        f"{free['lambda_err']:10.5f} "
        f"{pull:9.2f} "
        f"{free['chi2_red']:11.3f} "
        f"{fixed['chi2_red']:12.3f}"
    )


    selected_rows.append({
        "d": d,
        "w_expected": w_expected,
        "p_min": p_min,
        "p_max": p_max,
        "N": free["N"],
        "A": free["A"],
        "lambda": free["lambda"],
        "lambda_err": free["lambda_err"],
        "pull_sigma": pull,
        "chi2_free": free["chi2"],
        "chi2_red_free": free["chi2_red"],
        "A_fixed": fixed["A"],
        "chi2_fixed": fixed["chi2"],
        "chi2_red_fixed": fixed["chi2_red"],
    })


# =============================================================================
# Save results
# =============================================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


scan_df = pd.DataFrame(
    scan_rows
)


selected_df = pd.DataFrame(
    selected_rows
)


scan_df.to_csv(
    OUTPUT_SCAN,
    index=False,
)


selected_df.to_csv(
    OUTPUT_SELECTED,
    index=False,
)


print()
print("=" * 120)
print(f"Saved: {OUTPUT_SCAN}")
print(f"Saved: {OUTPUT_SELECTED}")
print("=" * 120)
