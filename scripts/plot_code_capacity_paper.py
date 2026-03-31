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


def majority_vote_curve(d: int, ps: np.ndarray) -> np.ndarray:
    n = d
    out = np.zeros_like(ps, dtype=float)
    for i, p in enumerate(ps):
        vals = []
        for k in range((n + 1) // 2, n + 1):
            vals.append(np.math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k)))
        out[i] = np.sum(vals)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--show-majority-vote", action="store_true")
    parser.add_argument("--xmin", type=float, default=None)
    parser.add_argument("--xmax", type=float, default=None)
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)
    args = parser.parse_args()

    df = load_summary(Path(args.csv))
    ds = sorted(df["distance"].unique())

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for i, d in enumerate(ds):
        sub = df[df["distance"] == d].sort_values("p_data")
        color = FOUR_COLORS[i % len(FOUR_COLORS)]

        ax.plot(
            sub["p_data"],
            sub["logical_error_rate"],
            linestyle="-",
            marker="o",
            markerfacecolor="None",
            color=color,
            label=str(d),
        )

        if args.show_majority_vote:
            xx = sub["p_data"].to_numpy()
            yy = majority_vote_curve(d, xx)
            ax.plot(xx, yy, "--", color="k")

    if args.threshold is not None:
        ax.axvline(args.threshold, linestyle="--", color="red")

    ax.grid()
    ax.set_xlabel(r"$p$")
    ax.set_ylabel(r"$p_L$")
    ax.set_xscale("log")
    ax.set_yscale("log")

    if args.xmin is not None or args.xmax is not None:
        ax.set_xlim(left=args.xmin, right=args.xmax)
    if args.ymin is not None or args.ymax is not None:
        ax.set_ylim(bottom=args.ymin, top=args.ymax)

    fig.legend(ds, title=r"$d$", ncol=len(ds), bbox_to_anchor=(0.12, 0.9, 0.8, 0.2))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
