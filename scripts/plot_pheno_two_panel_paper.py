#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
})

FOUR_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
SEVEN_COLORS = ["tab:blue", "lightsteelblue", "burlywood", "tab:orange", "gold", "yellowgreen", "olivedrab"]


def load_summary(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def add_stats(df: pd.DataFrame) -> pd.DataFrame:
    means = []
    errs = []
    for _, row in df.iterrows():
        y = np.load(row["data_file"])
        means.append(float(np.mean(y)))
        errs.append(float(np.std(y, ddof=0) / np.sqrt(len(y))))
    out = df.copy()
    out["mean_from_file"] = means
    out["stderr_from_file"] = errs
    return out


def select_best_reset(df: pd.DataFrame, xcol: str) -> pd.DataFrame:
    if "reset_time" in df.columns:
        df = df[df["reset_time"] <= df["distance"]].copy()
        idx = df.groupby(["distance", xcol])["mean_from_file"].idxmax()
        return df.loc[idx].copy().sort_values(["distance", xcol])
    return df.sort_values(["distance", xcol]).copy()


def choose_colors(n: int):
    return SEVEN_COLORS if n == 7 else FOUR_COLORS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-csv", required=True)
    parser.add_argument("--right-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--left-xcol", default="p_data")
    parser.add_argument("--right-xcol", default="p_meas")
    parser.add_argument("--left-xlabel", default=r"$p$")
    parser.add_argument("--right-xlabel", default=r"$q$")
    parser.add_argument("--xmin", type=float, default=1e-3)
    parser.add_argument("--xmax", type=float, default=1.0)
    parser.add_argument("--ymin", type=float, default=1.0)
    parser.add_argument("--ymax", type=float, default=1e8)
    args = parser.parse_args()

    left_df = select_best_reset(add_stats(load_summary(Path(args.left_csv))), args.left_xcol)
    right_df = select_best_reset(add_stats(load_summary(Path(args.right_csv))), args.right_xcol)

    ds = sorted(left_df["distance"].unique())
    colors = choose_colors(len(ds))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for i, d in enumerate(ds):
        sub = left_df[left_df["distance"] == d].sort_values(args.left_xcol)
        color = colors[i % len(colors)]
        ax[0].plot(
            sub[args.left_xcol],
            sub["mean_from_file"],
            "-",
            marker="o",
            markerfacecolor="None",
            color=color,
            label=str(d),
        )
        ax[0].errorbar(
            sub[args.left_xcol],
            sub["mean_from_file"],
            yerr=sub["stderr_from_file"],
            c="k",
            linestyle="none",
        )

    for i, d in enumerate(ds):
        sub = right_df[right_df["distance"] == d].sort_values(args.right_xcol)
        color = colors[i % len(colors)]
        ax[1].plot(
            sub[args.right_xcol],
            sub["mean_from_file"],
            "-",
            marker="o",
            markerfacecolor="None",
            color=color,
            label=str(d),
        )
        ax[1].errorbar(
            sub[args.right_xcol],
            sub["mean_from_file"],
            yerr=sub["stderr_from_file"],
            c="k",
            linestyle="none",
        )

    ax[0].grid()
    ax[1].grid()
    ax[0].set_ylabel(r"$\langle T_F \rangle$")
    ax[0].set_xlabel(args.left_xlabel)
    ax[1].set_xlabel(args.right_xlabel)
    ax[0].set_xscale("log")
    ax[1].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlim([args.xmin, args.xmax])
    ax[1].set_xlim([args.xmin, args.xmax])
    ax[0].set_ylim([args.ymin, args.ymax])

    fig.legend(ds, title=r"$d$", ncol=len(ds), bbox_to_anchor=(-0.182, 0.9, 1, 0.2))
    plt.subplots_adjust(wspace=0.15)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
