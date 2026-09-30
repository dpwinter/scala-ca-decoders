#!/usr/bin/env python3
"""
SCALA1D phenomenological noise:
pairwise finite-size crossing analysis.

Reads:
    data/pheno/pheno_pL_crossings_A.csv
    data/pheno/pheno_pL_crossings_B.csv
    data/pheno/pheno_pL_crossings_C.csv
    data/pheno/pheno_pL_crossings_D.csv
    data/pheno/pheno_pL_crossings_E.csv

For every neighboring simulated distance d1 < d2:

    1. Select resolved/stationary p=q points.
    2. Construct log(p_L) versus p curves.
    3. Find intersections within the common simulated p range using
       piecewise-linear interpolation.
    4. Plot the crossing p_x against sqrt(d1*d2).

Outputs:
    data/pheno/pheno_pairwise_crossings.csv
    figures/pheno_pairwise_crossings.png
    figures/pheno_pairwise_crossings.pdf

Run:
    python scripts/plot_pheno_pairwise_crossings.py
"""

from pathlib import Path
import glob
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================================
# Configuration
# =========================================================================

INPUT_GLOB = "data/pheno/pheno_pL_crossings_[A-E].csv"

OUTPUT_TABLE = Path(
    "data/pheno/pheno_pairwise_crossings.csv"
)

OUTPUT_PNG = Path(
    "figs/pheno_pairwise_crossings.png"
)

OUTPUT_PDF = Path(
    "figs/pheno_pairwise_crossings.pdf"
)

PQ_ATOL = 1e-12

REQUIRE_STATIONARY = True


# =========================================================================
# Load data
# =========================================================================

def load_data():

    files = sorted(glob.glob(INPUT_GLOB))

    if not files:
        raise FileNotFoundError(
            "No crossing datasets found.\n"
            f"Expected files matching:\n    {INPUT_GLOB}"
        )

    print("\nInput files:")

    frames = []

    for filename in files:

        print(f"  {filename}")

        frame = pd.read_csv(filename)

        frame["_source"] = filename

        frames.append(frame)

    df = pd.concat(
        frames,
        ignore_index=True
    )

    print(f"\nRaw rows: {len(df)}")
    print("Columns:")
    print(" ", list(df.columns))


    # ---------------------------------------------------------------------
    # Required columns
    # ---------------------------------------------------------------------

    required = {
        "d",
        "p",
        "q",
        "pL",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required CSV columns:\n    "
            + ", ".join(sorted(missing))
        )


    # ---------------------------------------------------------------------
    # Numerical conversion
    # ---------------------------------------------------------------------

    for column in [
        "d",
        "p",
        "q",
        "pL",
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


    if "pL_se" in df.columns:

        df["pL_se"] = pd.to_numeric(
            df["pL_se"],
            errors="coerce"
        )

    else:

        df["pL_se"] = np.nan


    # ---------------------------------------------------------------------
    # Stationarity flag
    # ---------------------------------------------------------------------

    if REQUIRE_STATIONARY and "stationary" in df.columns:

        if df["stationary"].dtype != bool:

            stationary = (
                df["stationary"]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            mapping = {
                "true": True,
                "1": True,
                "yes": True,
                "y": True,
                "false": False,
                "0": False,
                "no": False,
                "n": False,
            }

            df["stationary"] = stationary.map(mapping)

        before = len(df)

        df = df[
            df["stationary"] == True
        ].copy()

        print(
            f"Stationary filtering: "
            f"{before} -> {len(df)} rows"
        )


    # ---------------------------------------------------------------------
    # p=q only
    # ---------------------------------------------------------------------

    mask_pq = np.isclose(
        df["p"].to_numpy(dtype=float),
        df["q"].to_numpy(dtype=float),
        atol=PQ_ATOL,
        rtol=0.0
    )

    df = df[mask_pq].copy()


    # ---------------------------------------------------------------------
    # Basic validity
    # ---------------------------------------------------------------------

    mask = (
        np.isfinite(df["d"])
        & np.isfinite(df["p"])
        & np.isfinite(df["pL"])
        & (df["p"] > 0.0)
        & (df["pL"] > 0.0)
    )

    df = df[mask].copy()

    df["d"] = df["d"].astype(int)


    if df.empty:

        raise RuntimeError(
            "No usable p=q crossing data remain after filtering."
        )


    print(f"Usable rows: {len(df)}")

    return df


# =========================================================================
# Construct one p_L(p) curve
# =========================================================================

def curve_for_distance(df, d):

    x = df[
        df["d"] == d
    ].copy()

    if x.empty:
        return x


    # ---------------------------------------------------------------------
    # Handle duplicate (d,p) points
    #
    # These can occur if terminal batches overlap.
    #
    # If uncertainty exists, keep the run with the smallest pL_se.
    # Otherwise keep the first.
    # ---------------------------------------------------------------------

    if x["pL_se"].notna().any():

        x["_sort_se"] = x["pL_se"].fillna(np.inf)

        x = (
            x.sort_values(
                ["p", "_sort_se"]
            )
            .drop_duplicates(
                subset=["p"],
                keep="first"
            )
        )

        x = x.drop(
            columns=["_sort_se"]
        )

    else:

        x = (
            x.sort_values("p")
            .drop_duplicates(
                subset=["p"],
                keep="first"
            )
        )


    x = x.sort_values("p")

    return x.reset_index(drop=True)


# =========================================================================
# Interpolation
# =========================================================================

def interpolate_log_pL(curve, p):

    ps = curve["p"].to_numpy(dtype=float)

    log_y = np.log(
        curve["pL"].to_numpy(dtype=float)
    )

    p = np.asarray(
        p,
        dtype=float
    )


    if np.any(p < ps[0]) or np.any(p > ps[-1]):

        raise ValueError(
            "Interpolation attempted outside simulated p range."
        )


    return np.interp(
        p,
        ps,
        log_y
    )


# =========================================================================
# Crossing finder
# =========================================================================

def find_crossings(curve1, curve2):
    """
    Find all crossings of two curves inside their common p range.

    Each log(p_L) curve is piecewise linear in p.

    Therefore the difference

        Delta(p) =
            log p_L(d1,p)
            -
            log p_L(d2,p)

    is also piecewise linear on the union of their interpolation knots.
    """

    p1 = curve1[
        "p"
    ].to_numpy(dtype=float)

    p2 = curve2[
        "p"
    ].to_numpy(dtype=float)


    p_min = max(
        np.min(p1),
        np.min(p2)
    )

    p_max = min(
        np.max(p1),
        np.max(p2)
    )


    if p_max <= p_min:
        return []


    # ---------------------------------------------------------------------
    # Union of all interpolation knots inside common range
    # ---------------------------------------------------------------------

    knots1 = p1[
        (p1 >= p_min)
        & (p1 <= p_max)
    ]

    knots2 = p2[
        (p2 >= p_min)
        & (p2 <= p_max)
    ]

    knots = np.unique(
        np.concatenate([
            knots1,
            knots2,
            np.array([
                p_min,
                p_max
            ])
        ])
    )

    knots.sort()


    log1 = interpolate_log_pL(
        curve1,
        knots
    )

    log2 = interpolate_log_pL(
        curve2,
        knots
    )

    delta = log1 - log2


    crossings = []


    # ---------------------------------------------------------------------
    # Search every piecewise-linear interval
    # ---------------------------------------------------------------------

    for i in range(
        len(knots) - 1
    ):

        pa = knots[i]
        pb = knots[i + 1]

        da = delta[i]
        db = delta[i + 1]


        if not (
            np.isfinite(da)
            and np.isfinite(db)
        ):
            continue


        # Exact crossing at left knot
        if np.isclose(
            da,
            0.0,
            atol=1e-14,
            rtol=0.0
        ):

            crossings.append(
                float(pa)
            )

            continue


        # Sign change
        if da * db < 0.0:

            px = (
                pa
                - da
                * (pb - pa)
                / (db - da)
            )

            crossings.append(
                float(px)
            )


    # Exact crossing at final endpoint
    if np.isclose(
        delta[-1],
        0.0,
        atol=1e-14,
        rtol=0.0
    ):

        crossings.append(
            float(knots[-1])
        )


    # ---------------------------------------------------------------------
    # Remove numerical duplicates
    # ---------------------------------------------------------------------

    clean = []

    for px in crossings:

        if not clean:

            clean.append(px)

        elif not np.isclose(
            px,
            clean[-1],
            atol=1e-12,
            rtol=0.0
        ):

            clean.append(px)


    return clean


# =========================================================================
# Main
# =========================================================================

def main():

    df = load_data()


    # ---------------------------------------------------------------------
    # Available distances
    # ---------------------------------------------------------------------

    distances = sorted(
        df["d"].unique()
    )

    print("\nDistances present:")

    print(
        " ",
        " ".join(
            str(d)
            for d in distances
        )
    )


    # ---------------------------------------------------------------------
    # Construct curves
    # ---------------------------------------------------------------------

    curves = {}

    print("\nPoints per distance:")

    for d in distances:

        curve = curve_for_distance(
            df,
            d
        )

        if len(curve) < 2:

            print(
                f"  d={d:3d}: "
                f"{len(curve):3d} points "
                "(SKIPPED)"
            )

            continue


        curves[d] = curve

        print(
            f"  d={d:3d}: "
            f"{len(curve):3d} points, "
            f"p=["
            f"{curve['p'].min():.5f}, "
            f"{curve['p'].max():.5f}"
            f"]"
        )


    distances = sorted(
        curves.keys()
    )


    if len(distances) < 2:

        raise RuntimeError(
            "Need at least two usable distances."
        )


    # ---------------------------------------------------------------------
    # Neighboring-distance crossings
    # ---------------------------------------------------------------------

    results = []


    print("\nPairwise crossings:")
    print(
        "  d1   d2      sqrt(d1*d2)        p_x"
    )
    print(
        "  --   --      -----------        -------"
    )


    for d1, d2 in zip(
        distances[:-1],
        distances[1:]
    ):

        curve1 = curves[d1]
        curve2 = curves[d2]


        overlap_min = max(
            curve1["p"].min(),
            curve2["p"].min()
        )

        overlap_max = min(
            curve1["p"].max(),
            curve2["p"].max()
        )


        if overlap_max <= overlap_min:

            print(
                f"  {d1:3d}  {d2:3d}     "
                f"NO OVERLAP"
            )

            continue


        crossings = find_crossings(
            curve1,
            curve2
        )


        if len(crossings) == 0:

            print(
                f"  {d1:3d}  {d2:3d}     "
                f"NO CROSSING "
                f"(overlap "
                f"{overlap_min:.5f}-"
                f"{overlap_max:.5f})"
            )

            continue


        if len(crossings) > 1:

            crossing_string = ", ".join(
                f"{x:.6f}"
                for x in crossings
            )

            print(
                f"  {d1:3d}  {d2:3d}     "
                f"MULTIPLE CROSSINGS: "
                f"{crossing_string}"
            )

            # Do not arbitrarily choose one.
            continue


        px = crossings[0]

        d_eff = math.sqrt(
            d1 * d2
        )


        print(
            f"  {d1:3d}  {d2:3d}     "
            f"{d_eff:11.4f}        "
            f"{px:.7f}"
        )


        results.append({

            "d1": d1,

            "d2": d2,

            "d_eff": d_eff,

            "p_cross": px,

            "overlap_p_min":
                overlap_min,

            "overlap_p_max":
                overlap_max,

            "n_points_d1":
                len(curve1),

            "n_points_d2":
                len(curve2),

        })


    # ---------------------------------------------------------------------
    # Results table
    # ---------------------------------------------------------------------

    result = pd.DataFrame(
        results
    )


    if result.empty:

        raise RuntimeError(
            "No unique neighboring-distance crossings found."
        )


    result = (
        result
        .sort_values("d_eff")
        .reset_index(drop=True)
    )


    OUTPUT_TABLE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_TABLE,
        index=False
    )


    print("\nCrossing table:\n")

    print(
        result[
            [
                "d1",
                "d2",
                "d_eff",
                "p_cross"
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.7f}"
        )
    )


    print(
        f"\nSaved table:\n"
        f"  {OUTPUT_TABLE}"
    )


    # =====================================================================
    # Plot
    # =====================================================================

    x = result[
        "d_eff"
    ].to_numpy(dtype=float)

    y = result[
        "p_cross"
    ].to_numpy(dtype=float)


    fig, ax = plt.subplots(
        figsize=(8.0, 6.0)
    )


    ax.plot(
        x,
        y,
        "-o",
        linewidth=2.2,
        markersize=8,
    )


    # ---------------------------------------------------------------------
    # Labels
    # ---------------------------------------------------------------------

    ax.set_xlabel(
        r"Effective distance $\sqrt{d_1d_2}$",
        fontsize=18,
        labelpad=10
    )


    ax.set_ylabel(
        r"Finite-size crossover $p_{\times}$",
        fontsize=18,
        labelpad=10
    )


    # ---------------------------------------------------------------------
    # Axes
    # ---------------------------------------------------------------------

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=15,
        width=1.4,
        length=6
    )


    for spine in ax.spines.values():

        spine.set_linewidth(
            1.4
        )


    # Match the old plot convention.
    ymax = max(
        0.0365,
        1.06 * np.max(y)
    )

    ax.set_ylim(
        0.0,
        ymax
    )


    if len(x) > 1:

        xspan = (
            np.max(x)
            - np.min(x)
        )

        ax.set_xlim(
            np.min(x)
            - 0.05 * xspan,
            np.max(x)
            + 0.05 * xspan
        )


    ax.grid(False)


    # ---------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------

    fig.tight_layout()


    OUTPUT_PNG.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    fig.savefig(
        OUTPUT_PNG,
        dpi=300,
        bbox_inches="tight"
    )


    fig.savefig(
        OUTPUT_PDF,
        bbox_inches="tight"
    )


    print(
        f"\nSaved figures:\n"
        f"  {OUTPUT_PNG}\n"
        f"  {OUTPUT_PDF}"
    )


    plt.show()


if __name__ == "__main__":
    main()
