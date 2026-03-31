#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "distance" not in df.columns or "p_data" not in df.columns:
        raise ValueError(f"{path} is missing required columns")
    return df


def plot_code_capacity(ax, df: pd.DataFrame) -> None:
    if "logical_error_rate" not in df.columns:
        raise ValueError("Code-capacity summary is missing 'logical_error_rate'")

    df = df.sort_values(["distance", "p_data"])
    for d in sorted(df["distance"].unique()):
        sub = df[df["distance"] == d].sort_values("p_data")
        ax.plot(
            sub["p_data"],
            sub["logical_error_rate"],
            marker="o",
            label=f"d={d}",
        )

    ax.set_xlabel("data noise p")
    ax.set_ylabel(r"$p_L$")
    ax.set_title("Harrington2D code capacity")
    ax.grid(True)
    ax.legend()


def plot_phenomenological(ax, df: pd.DataFrame) -> None:
    if "mean_lifetime" not in df.columns:
        raise ValueError("Phenomenological summary is missing 'mean_lifetime'")

    df = df.sort_values(["distance", "p_data"])
    for d in sorted(df["distance"].unique()):
        sub = df[df["distance"] == d].sort_values("p_data")
        ax.plot(
            sub["p_data"],
            sub["mean_lifetime"],
            marker="o",
            label=f"d={d}",
        )

    ax.set_xlabel("data noise p")
    ax.set_ylabel(r"$\langle T_F \rangle$")
    ax.set_title("Harrington2D phenomenological")
    ax.grid(True)
    ax.legend()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc-csv", required=True, help="Summary CSV for code-capacity scan")
    parser.add_argument("--pheno-csv", required=True, help="Summary CSV for phenomenological scan")
    parser.add_argument("--output-dir", required=True, help="Directory for output figures")
    parser.add_argument("--logx", action="store_true", help="Use logarithmic x-axis")
    parser.add_argument("--logy-cc", action="store_true", help="Use logarithmic y-axis for code capacity")
    parser.add_argument("--logy-pheno", action="store_true", help="Use logarithmic y-axis for phenomenological")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    cc_df = load_summary(Path(args.cc_csv))
    ph_df = load_summary(Path(args.pheno_csv))

    cc_df = cc_df[cc_df["model"] == "harrington2d"]
    ph_df = ph_df[ph_df["model"] == "harrington2d"]

    fig1, ax1 = plt.subplots(figsize=(6, 4.5))
    plot_code_capacity(ax1, cc_df)
    if args.logx:
        ax1.set_xscale("log")
    if args.logy_cc:
        ax1.set_yscale("log")
    fig1.tight_layout()
    cc_path = outdir / "harrington2d_code_capacity.png"
    fig1.savefig(cc_path, dpi=200)

    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    plot_phenomenological(ax2, ph_df)
    if args.logx:
        ax2.set_xscale("log")
    if args.logy_pheno:
        ax2.set_yscale("log")
    fig2.tight_layout()
    ph_path = outdir / "harrington2d_phenomenological.png"
    fig2.savefig(ph_path, dpi=200)

    print(f"Wrote {cc_path}")
    print(f"Wrote {ph_path}")


if __name__ == "__main__":
    main()
