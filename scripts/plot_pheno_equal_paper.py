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


def select_best_reset(df: pd.DataFrame) -> pd.DataFrame:
    if "reset_time" in df.columns:
        df = df[df["reset_time"] <= df["distance"]].copy()
        idx = df.groupby(["distance", "p_data"])["mean_from_file"].idxmax()
        return df.loc[idx].copy().sort_values(["distance", "p_data"])
    return df.sort_values(["distance", "p_data"]).copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--xlabel", default=r"$p = q = p_{\mathrm{sig}}$")
    parser.add_argument("--xmin", type=float, default=1e-3)
    parser.add_argument("--xmax", type=float, default=1.0)
    parser.add_argument("--ymin", type=float, default=1.0)
    parser.add_argument("--ymax", type=float, default=1e9)
    args = parser.parse_args()

    df = add_stats(load_summary(Path(args.csv)))
    df = select_best_reset(df)

    ds = sorted(df["distance"].unique())

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for i, d in enumerate(ds):
        sub = df[df["distance"] == d].sort_values("p_data")
        color = FOUR_COLORS[i % len(FOUR_COLORS)]
        ax.plot(
            sub["p_data"],
            sub["mean_from_file"],
            "-",
            marker="o",
            markerfacecolor="None",
            color=color,
            label=str(d),
        )
        ax.errorbar(
            sub["p_data"],
            sub["mean_from_file"],
            yerr=sub["stderr_from_file"],
            c="k",
            linestyle="none",
        )

    ax.grid()
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(r"$\langle T_F \rangle$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([args.xmin, args.xmax])
    ax.set_ylim([args.ymin, args.ymax])

    fig.legend(ds, title=r"$d$", ncol=len(ds), bbox_to_anchor=(0.12, 0.9, 0.8, 0.2))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
