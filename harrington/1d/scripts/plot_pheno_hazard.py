from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED = {
    "d",
    "p",
    "q",
    "t_start",
    "t_end",
    "shots",
    "at_risk",
    "failures",
    "exposure",
}


# =============================================================================
# Load
# =============================================================================

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path
    ).copy()

    missing = (
        REQUIRED
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    for col in REQUIRED:
        df[col] = pd.to_numeric(
            df[col],
            errors="raise",
        )

    return df


# =============================================================================
# Hazard curve
# =============================================================================

def hazard_curve(
    df: pd.DataFrame,
    d: int,
    p: float,
    xmax: float,
    min_at_risk: int,
    min_risk_fraction: float,
) -> pd.DataFrame:

    sub = df[
        (df["d"] == d)
        & np.isclose(
            df["p"],
            p,
            atol=1e-12,
            rtol=0.0,
        )
        & np.isclose(
            df["q"],
            p,
            atol=1e-12,
            rtol=0.0,
        )
    ].copy()

    if sub.empty:
        raise RuntimeError(
            f"No data for d={d}, p=q={p}"
        )

    sub = sub[
        sub["t_start"] < xmax
    ].copy()

    # Multiple independent simulation batches are combined exactly
    # through additive sufficient statistics.
    grouped = (
        sub.groupby(
            [
                "t_start",
                "t_end",
            ],
            as_index=False,
            sort=True,
        )
        .agg(
            {
                "shots": "sum",
                "at_risk": "sum",
                "failures": "sum",
                "exposure": "sum",
            }
        )
    )

    rows = []

    for _, row in grouped.iterrows():

        shots = int(
            row["shots"]
        )

        at_risk = int(
            row["at_risk"]
        )

        failures = int(
            row["failures"]
        )

        exposure = float(
            row["exposure"]
        )

        risk_cutoff = max(
            int(min_at_risk),
            int(
                math.ceil(
                    min_risk_fraction
                    * shots
                )
            ),
        )

        if at_risk < risk_cutoff:
            break

        if (
            failures <= 0
            or exposure <= 0
        ):
            continue

        hazard = (
            failures
            / exposure
        )

        hazard_err = (
            math.sqrt(failures)
            / exposure
        )

        rows.append(
            {
                "x":
                    0.5
                    * (
                        float(
                            row["t_start"]
                        )
                        + float(
                            row["t_end"]
                        )
                    ),
                "hazard":
                    hazard,
                "hazard_err":
                    hazard_err,
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Panel
# =============================================================================

def plot_panel(
    ax,
    df,
    specs,
    xmax,
    min_at_risk,
    min_risk_fraction,
    legend_title,
    legend_location,
):

    colors = plt.rcParams[
        "axes.prop_cycle"
    ].by_key()["color"]

    for i, (
        d,
        p,
        label,
    ) in enumerate(specs):

        curve = hazard_curve(
            df=df,
            d=d,
            p=p,
            xmax=xmax,
            min_at_risk=min_at_risk,
            min_risk_fraction=(
                min_risk_fraction
            ),
        )

        if curve.empty:
            print(
                f"WARNING: no usable bins "
                f"for d={d}, p=q={p}"
            )
            continue

        color = colors[
            i % len(colors)
        ]

        ax.errorbar(
            curve["x"],
            curve["hazard"],
            yerr=curve[
                "hazard_err"
            ],

            fmt="o-",

            color=color,

            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=1.0,

            markersize=3.8,
            linewidth=1.1,

            capsize=1.7,
            elinewidth=0.75,

            label=label,
            zorder=3,
        )

    ax.set_yscale(
        "log"
    )

    ax.set_xlim(
        0,
        xmax,
    )

    ax.set_xlabel(
        r"time $t$",
        fontsize=9.5,
        labelpad=2,
    )

    ax.grid(
        True,
        which="major",
        alpha=0.35,
        linewidth=0.7,
    )

    ax.grid(
        True,
        which="minor",
        alpha=0.10,
        linewidth=0.4,
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=8.5,
        pad=2,
    )

    ax.tick_params(
        axis="both",
        which="minor",
        length=2,
    )

    leg = ax.legend(
        title=legend_title,
        loc=legend_location,
        frameon=True,
        fontsize=8,
        title_fontsize=8.5,
        borderpad=0.5,
        handlelength=1.8,
        labelspacing=0.3,
        handletextpad=0.5,
    )

    leg.get_frame().set_alpha(
        0.88
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/pheno/"
            "pheno_hazard_scan.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "figs/pheno_hazard"
        ),
    )

    parser.add_argument(
        "--fixed-p",
        type=float,
        default=0.015,
    )

    parser.add_argument(
        "--distances",
        nargs="+",
        type=int,
        default=[
            3,
            9,
            27,
            81,
        ],
    )

    parser.add_argument(
        "--left-xmax",
        type=float,
        default=1500.0,
    )

    parser.add_argument(
        "--fixed-d",
        type=int,
        default=27,
    )

    parser.add_argument(
        "--p-values",
        nargs="+",
        type=float,
        default=[
            0.015,
            0.020,
            0.025,
            0.030,
        ],
    )

    parser.add_argument(
        "--right-xmax",
        type=float,
        default=1000.0,
    )

    parser.add_argument(
        "--min-at-risk",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--min-risk-fraction",
        type=float,
        default=0.01,
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():

    args = parse_args()

    df = load_data(
        args.input
    )

    left_specs = [
        (
            d,
            args.fixed_p,
            rf"${d}$",
        )
        for d in args.distances
    ]

    right_specs = [
        (
            args.fixed_d,
            p,
            rf"${p:.3f}$",
        )
        for p in args.p_values
    ]

    fig, (
        ax_left,
        ax_right,
    ) = plt.subplots(
        1,
        2,
        figsize=(
            9.2,
            2.65,
        ),
        sharey=True,
    )

    plot_panel(
        ax=ax_left,
        df=df,
        specs=left_specs,
        xmax=args.left_xmax,
        min_at_risk=(
            args.min_at_risk
        ),
        min_risk_fraction=(
            args.min_risk_fraction
        ),
        legend_title=r"$d$",
        legend_location="lower right",
    )

    plot_panel(
        ax=ax_right,
        df=df,
        specs=right_specs,
        xmax=args.right_xmax,
        min_at_risk=(
            args.min_at_risk
        ),
        min_risk_fraction=(
            args.min_risk_fraction
        ),
        legend_title=r"$p=q$",
        legend_location="lower right",
    )

    ax_left.set_ylabel(
        r"logical hazard $h(t)$",
        fontsize=9.5,
        labelpad=2,
    )

    ax_left.text(
        0.50,
        0.045,
        rf"$p=q={args.fixed_p:.3f}$",
        transform=ax_left.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.8,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.78,
            "pad": 1.5,
        },
        zorder=10,
    )

    ax_right.text(
        0.50,
        0.045,
        rf"$d={args.fixed_d}$",
        transform=ax_right.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.8,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.78,
            "pad": 1.5,
        },
        zorder=10,
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.23,
        top=0.985,
        wspace=0.10,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        args.output.with_suffix(
            ".pdf"
        ),
        bbox_inches="tight",
        pad_inches=0.02,
    )

    fig.savefig(
        args.output.with_suffix(
            ".png"
        ),
        dpi=350,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(
        fig
    )


if __name__ == "__main__":
    main()
