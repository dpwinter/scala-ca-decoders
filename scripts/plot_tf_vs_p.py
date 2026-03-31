#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"model", "distance", "p_data", "mean_lifetime"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df


def reduce_scala(df: pd.DataFrame) -> pd.DataFrame:
    """
    For SCALA1D, keep the best lifetime over reset_time at fixed (distance, p_data).
    """
    group_cols = ["distance", "p_data"]
    idx = df.groupby(group_cols)["mean_lifetime"].idxmax()
    return df.loc[idx].copy().sort_values(group_cols)


def reduce_harrington(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harrington scan already uses fixed decoder parameters.
    """
    return df.copy().sort_values(["distance", "p_data"])


def plot_model(ax, df: pd.DataFrame, model_name: str) -> None:
    for d in sorted(df["distance"].unique()):
        sub = df[df["distance"] == d].sort_values("p_data")
        ax.plot(
            sub["p_data"],
            sub["mean_lifetime"],
            marker="o",
            label=f"{model_name}, d={d}",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scala-csv", required=True, help="Summary CSV for scala1d phenomenological scan")
    parser.add_argument("--harrington-csv", required=True, help="Summary CSV for harrington1d phenomenological scan")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--logx", action="store_true")
    parser.add_argument("--logy", action="store_true")
    args = parser.parse_args()

    scala_df = load_summary(Path(args.scala_csv))
    harr_df = load_summary(Path(args.harrington_csv))

    scala_df = scala_df[scala_df["model"] == "scala1d"]
    harr_df = harr_df[harr_df["model"] == "harrington1d"]

    scala_df = reduce_scala(scala_df)
    harr_df = reduce_harrington(harr_df)

    fig, ax = plt.subplots(figsize=(7, 5))

    plot_model(ax, scala_df, "SCALA1D")
    plot_model(ax, harr_df, "Harrington1D")

    ax.set_xlabel("data noise p")
    ax.set_ylabel(r"$\langle T_F \rangle$")
    ax.legend()

    if args.logx:
        ax.set_xscale("log")
    if args.logy:
        ax.set_yscale("log")

    ax.grid(True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)

    print(f"Wrote plot to {output_path}")


if __name__ == "__main__":
    main()
