from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/pheno/harrington1d_tf_hazard_summary.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "figs/tf_hazard_check/"
            "hazard_times_mean_tf_by_distance.png"
        ),
    )

    args = parser.parse_args()

    df = pd.read_csv(
        args.input
    )

    df = df[
        df["stationary"]
    ].copy()

    if len(df) == 0:
        raise RuntimeError(
            "No stationary groups found."
        )

    distances = sorted(
        df["d"].unique()
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4),
        sharey=True,
    )

    # =========================================================================
    # Left: data noise only
    # =========================================================================

    data_df = df[
        df["noise_mode"] == "data"
    ]

    for d in distances:

        g = data_df[
            data_df["d"] == d
        ].sort_values(
            "inv_mean_TF"
        )

        if len(g) == 0:
            continue

        axes[0].errorbar(
            g["inv_mean_TF"],
            g["R"],
            yerr=[
                (
                    g["R"]
                    - g["R_lo"]
                ).to_numpy(),
                (
                    g["R_hi"]
                    - g["R"]
                ).to_numpy(),
            ],
            marker="o",
            markerfacecolor="none",
            linestyle="-",
            capsize=2,
            label=rf"$d={d}$",
        )

    axes[0].axhline(
        1.0,
        linestyle="--",
    )

    axes[0].set_xscale(
        "log"
    )

    axes[0].set_xlabel(
        r"$1/\langle T_F\rangle$"
    )

    axes[0].set_ylabel(
        r"$h_\infty\langle T_F\rangle$"
    )

    axes[0].set_title(
        "data noise"
    )

    axes[0].legend()

    # =========================================================================
    # Right: measurement noise only
    # =========================================================================

    meas_df = df[
        df["noise_mode"] == "measurement"
    ]

    for d in distances:

        g = meas_df[
            meas_df["d"] == d
        ].sort_values(
            "inv_mean_TF"
        )

        if len(g) == 0:
            continue

        axes[1].errorbar(
            g["inv_mean_TF"],
            g["R"],
            yerr=[
                (
                    g["R"]
                    - g["R_lo"]
                ).to_numpy(),
                (
                    g["R_hi"]
                    - g["R"]
                ).to_numpy(),
            ],
            marker="o",
            markerfacecolor="none",
            linestyle="-",
            capsize=2,
            label=rf"$d={d}$",
        )

    axes[1].axhline(
        1.0,
        linestyle="--",
    )

    axes[1].set_xscale(
        "log"
    )

    axes[1].set_xlabel(
        r"$1/\langle T_F\rangle$"
    )

    axes[1].set_title(
        "measurement noise"
    )

    axes[1].legend()

    # =========================================================================
    # Shared style
    # =========================================================================

    for ax in axes:

        ax.grid(
            True,
            which="major",
        )

        ax.grid(
            False,
            which="minor",
        )

    fig.tight_layout()

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        args.output,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        args.output.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    # =========================================================================
    # Print numerical summary per distance
    # =========================================================================

    print()
    print("=" * 100)
    print(
        "HARRINGTON1D: h_inf <T_F> BY DISTANCE"
    )
    print("=" * 100)

    print(
        f"{'mode':>12s} "
        f"{'d':>5s} "
        f"{'N':>6s} "
        f"{'median R':>12s} "
        f"{'median |R-1|':>16s} "
        f"{'min R':>10s} "
        f"{'max R':>10s}"
    )

    print("-" * 100)

    for mode in [
        "data",
        "measurement",
    ]:

        for d in distances:

            g = df[
                (
                    df["noise_mode"]
                    == mode
                )
                & (
                    df["d"]
                    == d
                )
            ]

            if len(g) == 0:
                continue

            R = g[
                "R"
            ].to_numpy()

            print(
                f"{mode:>12s} "
                f"{d:5d} "
                f"{len(g):6d} "
                f"{np.nanmedian(R):12.4f} "
                f"{np.nanmedian(np.abs(R - 1.0)):16.4f} "
                f"{np.nanmin(R):10.4f} "
                f"{np.nanmax(R):10.4f}"
            )

    print("=" * 100)

    plt.show()


if __name__ == "__main__":
    main()
