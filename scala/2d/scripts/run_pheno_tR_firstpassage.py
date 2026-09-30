from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the SCALA2D first-passage simulation that we have already tested.
from scripts.run_pheno_hazard import simulate_point


# =============================================================================
# Reset-period grid
# =============================================================================

def make_reset_periods(
    d: int,
    alpha_min: float,
    alpha_max: float,
) -> list[int]:
    """
    Use every distinct integer reset period in the requested normalized range.

    This avoids duplicate simulations caused by rounding alpha*d.
    """

    tR_min = max(
        1,
        int(
            math.ceil(
                alpha_min * d
            )
        ),
    )

    tR_max = max(
        tR_min,
        int(
            math.floor(
                alpha_max * d
            )
        ),
    )

    return list(
        range(
            tR_min,
            tR_max + 1,
        )
    )


# =============================================================================
# Plateau estimator
# =============================================================================

def aggregate_interval(
    df: pd.DataFrame,
):
    failures = int(
        df["failures"].sum()
    )

    exposure = float(
        df["exposure"].sum()
    )

    if (
        failures <= 0
        or exposure <= 0
    ):
        return (
            np.nan,
            np.nan,
            failures,
            exposure,
        )

    h = (
        failures
        / exposure
    )

    se = (
        math.sqrt(
            failures
        )
        / exposure
    )

    return (
        h,
        se,
        failures,
        exposure,
    )


def weighted_slope_test(
    sub: pd.DataFrame,
):
    """
    Weighted linear fit

        h(t) = a + b t

    using approximate Poisson hazard uncertainties.

    Returns
        slope,
        SE_slope,
        z_slope
    """

    failures = (
        sub["failures"]
        .to_numpy(
            dtype=float
        )
    )

    exposure = (
        sub["exposure"]
        .to_numpy(
            dtype=float
        )
    )

    x = (
        0.5
        * (
            sub["t_start"].to_numpy(
                dtype=float
            )
            + sub["t_end"].to_numpy(
                dtype=float
            )
        )
    )

    good = (
        (failures > 0)
        & (exposure > 0)
    )

    if np.count_nonzero(
        good
    ) < 3:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    x = x[good]

    y = (
        failures[good]
        / exposure[good]
    )

    sigma = (
        np.sqrt(
            failures[good]
        )
        / exposure[good]
    )

    # Avoid singular weights.
    sigma = np.maximum(
        sigma,
        1e-300,
    )

    w = (
        1.0
        / sigma**2
    )

    X = np.column_stack(
        [
            np.ones_like(x),
            x,
        ]
    )

    XT_W = (
        X.T
        * w
    )

    normal = (
        XT_W
        @ X
    )

    try:
        cov = np.linalg.inv(
            normal
        )
    except np.linalg.LinAlgError:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    beta = (
        cov
        @ XT_W
        @ y
    )

    slope = float(
        beta[1]
    )

    se_slope = math.sqrt(
        max(
            float(
                cov[1, 1]
            ),
            0.0,
        )
    )

    if se_slope <= 0:
        z = np.nan
    else:
        z = (
            slope
            / se_slope
        )

    return (
        slope,
        se_slope,
        z,
    )


def half_consistency_test(
    sub: pd.DataFrame,
):
    """
    Compare hazards in the first and second halves of a candidate plateau.
    """

    n = len(
        sub
    )

    if n < 4:
        return np.nan

    mid = (
        n // 2
    )

    first = sub.iloc[
        :mid
    ]

    second = sub.iloc[
        mid:
    ]

    (
        h1,
        se1,
        _,
        _,
    ) = aggregate_interval(
        first
    )

    (
        h2,
        se2,
        _,
        _,
    ) = aggregate_interval(
        second
    )

    if (
        not np.isfinite(h1)
        or not np.isfinite(h2)
        or not np.isfinite(se1)
        or not np.isfinite(se2)
    ):
        return np.nan

    denom = math.sqrt(
        se1**2
        + se2**2
    )

    if denom <= 0:
        return np.nan

    return (
        h1 - h2
    ) / denom


def detect_plateau(
    rows: list[dict],
    min_at_risk: int,
    min_risk_fraction: float,
    min_bins: int,
    min_events: int,
    max_slope_z: float,
    max_half_z: float,
):
    """
    Find the earliest statistically acceptable stationary suffix.

    Procedure:
      1. discard late bins once the survivor population becomes too small;
      2. scan candidate plateau starts from early to late;
      3. require:
           - enough bins,
           - enough failures,
           - no significant linear slope,
           - no significant difference between first/second half;
      4. choose the earliest candidate satisfying all tests.
    """

    df = pd.DataFrame(
        rows
    ).copy()

    if df.empty:
        return None

    df = df.sort_values(
        "t_start"
    ).reset_index(
        drop=True
    )

    shots = int(
        df["shots"].iloc[0]
    )

    risk_cutoff = max(
        int(
            min_at_risk
        ),
        int(
            math.ceil(
                min_risk_fraction
                * shots
            )
        ),
    )

    usable = []

    for _, row in df.iterrows():

        if int(
            row["at_risk"]
        ) < risk_cutoff:
            break

        usable.append(
            row
        )

    if len(
        usable
    ) < min_bins:
        return None

    usable = pd.DataFrame(
        usable
    ).reset_index(
        drop=True
    )

    # -------------------------------------------------------------------------
    # Earliest acceptable plateau
    # -------------------------------------------------------------------------

    for start in range(
        0,
        len(usable)
        - min_bins
        + 1,
    ):

        sub = usable.iloc[
            start:
        ]

        (
            h,
            se,
            nfail,
            exposure,
        ) = aggregate_interval(
            sub
        )

        if (
            not np.isfinite(h)
            or nfail < min_events
        ):
            continue

        (
            slope,
            slope_se,
            slope_z,
        ) = weighted_slope_test(
            sub
        )

        half_z = (
            half_consistency_test(
                sub
            )
        )

        if not np.isfinite(
            slope_z
        ):
            continue

        if not np.isfinite(
            half_z
        ):
            continue

        if (
            abs(slope_z)
            <= max_slope_z
            and abs(half_z)
            <= max_half_z
        ):

            return {
                "plateau_start":
                    float(
                        sub[
                            "t_start"
                        ].iloc[0]
                    ),

                "plateau_end":
                    float(
                        sub[
                            "t_end"
                        ].iloc[-1]
                    ),

                "h_inf":
                    float(
                        h
                    ),

                "SE_h_inf":
                    float(
                        se
                    ),

                "plateau_failures":
                    int(
                        nfail
                    ),

                "plateau_exposure":
                    float(
                        exposure
                    ),

                "slope":
                    float(
                        slope
                    ),

                "SE_slope":
                    float(
                        slope_se
                    ),

                "slope_z":
                    float(
                        slope_z
                    ),

                "half_z":
                    float(
                        half_z
                    ),

                "risk_cutoff":
                    int(
                        risk_cutoff
                    ),
            }

    return None


# =============================================================================
# Main
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--d",
        nargs="+",
        type=int,
        default=[
            9,
            21,
        ],
    )

    parser.add_argument(
        "--p",
        type=float,
        default=0.004,
    )

    parser.add_argument(
        "--q",
        type=float,
        default=None,
        help="Default: q=p.",
    )

    parser.add_argument(
        "--alpha-min",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--alpha-max",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--shots",
        type=int,
        default=20000,
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=2000,
        help=(
            "Maximum first-passage runtime "
            "in raw CA/QEC rounds."
        ),
    )

    parser.add_argument(
        "--bin-width",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--walker-chunk",
        type=int,
        default=1024,
    )

    # -------------------------------------------------------------------------
    # Plateau criteria
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--min-at-risk",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--min-risk-fraction",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--min-plateau-bins",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--min-plateau-events",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--max-slope-z",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--max-half-z",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
    )

    parser.add_argument(
        "--timeseries-output",
        type=Path,
        default=Path(
            "data/pheno/"
            "pheno_tR_firstpassage_timeseries.csv"
        ),
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "data/pheno/"
            "pheno_tR_firstpassage_summary.csv"
        ),
    )

    args = parser.parse_args()

    q = (
        args.p
        if args.q is None
        else args.q
    )

    all_rows = []
    summary_rows = []

    points = []

    for d in args.d:

        periods = make_reset_periods(
            d=d,
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
        )

        for tR in periods:

            points.append(
                (
                    d,
                    tR,
                )
            )

    print()
    print("=" * 110)
    print(
        "SCALA2D FIRST-PASSAGE RESET-PERIOD SCAN"
    )
    print("=" * 110)

    print(
        f"p                  = {args.p}"
    )

    print(
        f"q                  = {q}"
    )

    print(
        f"shots / point      = {args.shots}"
    )

    print(
        f"runtime             = {args.rounds}"
    )

    print(
        f"bin width           = {args.bin_width}"
    )

    print(
        f"points              = {len(points)}"
    )

    print("-" * 110)

    for j, (
        d,
        tR,
    ) in enumerate(
        points,
        start=1,
    ):

        alpha = (
            tR / d
        )

        point_seed = (
            args.seed
            + 10007 * j
            + 101 * d
            + 100000 * tR
        )

        print(
            f"[{j:2d}/{len(points):2d}] "
            f"d={d:3d}  "
            f"p=q={args.p:.5f}  "
            f"tR={tR:3d}  "
            f"tR/d={alpha:.5f}"
        )

        # ---------------------------------------------------------------------
        # Reuse the validated first-passage hazard simulation.
        #
        # Passing reset_factor=tR/d gives exactly this integer reset period.
        # ---------------------------------------------------------------------

        rows = simulate_point(
            d=d,
            p=args.p,
            q=q,
            shots=args.shots,
            rounds=args.rounds,
            bin_width=args.bin_width,
            walker_chunk=args.walker_chunk,
            reset_factor=alpha,
            seed=point_seed,
        )

        # Add explicit scan metadata.
        for row in rows:

            row[
                "requested_reset_period"
            ] = int(
                tR
            )

            row[
                "actual_reset_factor"
            ] = float(
                tR / d
            )

        all_rows.extend(
            rows
        )

        plateau = detect_plateau(
            rows=rows,
            min_at_risk=args.min_at_risk,
            min_risk_fraction=(
                args.min_risk_fraction
            ),
            min_bins=(
                args.min_plateau_bins
            ),
            min_events=(
                args.min_plateau_events
            ),
            max_slope_z=(
                args.max_slope_z
            ),
            max_half_z=(
                args.max_half_z
            ),
        )

        if plateau is None:

            print(
                "        plateau: NOT FOUND"
            )

            summary_rows.append(
                {
                    "d":
                        int(d),

                    "p":
                        float(
                            args.p
                        ),

                    "q":
                        float(
                            q
                        ),

                    "reset_period":
                        int(
                            tR
                        ),

                    "actual_reset_factor":
                        float(
                            alpha
                        ),

                    "h_inf":
                        np.nan,

                    "SE_h_inf":
                        np.nan,

                    "plateau_start":
                        np.nan,

                    "plateau_end":
                        np.nan,

                    "plateau_failures":
                        0,

                    "plateau_exposure":
                        0.0,

                    "slope_z":
                        np.nan,

                    "half_z":
                        np.nan,

                    "plateau_ok":
                        False,
                }
            )

            continue

        print(
            f"        plateau "
            f"[{plateau['plateau_start']:.0f}, "
            f"{plateau['plateau_end']:.0f}]  "
            f"h={plateau['h_inf']:.6e}  "
            f"SE={plateau['SE_h_inf']:.2e}  "
            f"Nfail={plateau['plateau_failures']:6d}  "
            f"z_slope={plateau['slope_z']:+.2f}  "
            f"z_half={plateau['half_z']:+.2f}"
        )

        summary_rows.append(
            {
                "d":
                    int(d),

                "p":
                    float(
                        args.p
                    ),

                "q":
                    float(
                        q
                    ),

                "reset_period":
                    int(
                        tR
                    ),

                "actual_reset_factor":
                    float(
                        alpha
                    ),

                "h_inf":
                    float(
                        plateau[
                            "h_inf"
                        ]
                    ),

                "SE_h_inf":
                    float(
                        plateau[
                            "SE_h_inf"
                        ]
                    ),

                "plateau_start":
                    float(
                        plateau[
                            "plateau_start"
                        ]
                    ),

                "plateau_end":
                    float(
                        plateau[
                            "plateau_end"
                        ]
                    ),

                "plateau_failures":
                    int(
                        plateau[
                            "plateau_failures"
                        ]
                    ),

                "plateau_exposure":
                    float(
                        plateau[
                            "plateau_exposure"
                        ]
                    ),

                "slope_z":
                    float(
                        plateau[
                            "slope_z"
                        ]
                    ),

                "half_z":
                    float(
                        plateau[
                            "half_z"
                        ]
                    ),

                "plateau_ok":
                    True,
            }
        )

    # =========================================================================
    # Save
    # =========================================================================

    args.timeseries_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    timeseries_df = pd.DataFrame(
        all_rows
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    append_replace(
        path=args.timeseries_output,
        new_df=timeseries_df,
        key_cols=[
            "d",
            "p",
            "q",
            "requested_reset_period",
        ],
    )

    append_replace(
        path=args.summary_output,
        new_df=summary_df,
        key_cols=[
            "d",
            "p",
            "q",
            "reset_period",
        ],
    )

    print("-" * 110)

    print(
        f"Saved time series: "
        f"{args.timeseries_output}"
    )

    print(
        f"Saved summary:     "
        f"{args.summary_output}"
    )

    print("=" * 110)

def append_replace(
    path: Path,
    new_df: pd.DataFrame,
    key_cols: list[str],
):
    """
    Append new parameter points to an existing CSV.

    If a parameter point already exists, replace the old rows for that
    point rather than duplicating them.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not path.exists():
        new_df.to_csv(
            path,
            index=False,
        )
        return

    old_df = pd.read_csv(
        path
    )

    if old_df.empty:
        new_df.to_csv(
            path,
            index=False,
        )
        return

    # Build tuples identifying all parameter points being replaced.
    new_keys = set(
        tuple(row)
        for row in new_df[
            key_cols
        ].itertuples(
            index=False,
            name=None,
        )
    )

    keep = []

    for row in old_df[
        key_cols
    ].itertuples(
        index=False,
        name=None,
    ):
        keep.append(
            tuple(row)
            not in new_keys
        )

    old_df = old_df[
        np.asarray(
            keep,
            dtype=bool,
        )
    ]

    out = pd.concat(
        [
            old_df,
            new_df,
        ],
        ignore_index=True,
    )

    out.to_csv(
        path,
        index=False,
    )


if __name__ == "__main__":
    main()
