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


COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


def load_summary(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def fit_lambda(sub: pd.DataFrame, nfit: int = 4) -> float:
    s = sub.sort_values("p_data")
    s = s[(s["logical_error_rate"] > 0) & (s["logical_error_rate"] < 1)]
    if len(s) < 2:
        return np.nan
    s = s.iloc[: min(nfit, len(s))]
    x = np.log(s["p_data"].to_numpy())
    y = np.log(s["logical_error_rate"].to_numpy())
    m, _ = np.polyfit(x, y, 1)
    return float(m)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Summary CSV for SCALA2D code-capacity scan")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--threshold", type=float, default=None, help="Optional vertical threshold line")
    args = parser.parse_args()

    df = load_summary(Path(args.csv))
    df = df[df["model"] == "scala2d"].copy()

    ds = sorted(df["distance"].unique())

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for i, d in enumerate(ds):
        sub = df[df["distance"] == d].sort_values("p_data")
        color = COLORS[i % len(COLORS)]
        ax.plot(
            sub["p_data"],
            sub["logical_error_rate"],
            linestyle="-",
            marker="o",
            markerfacecolor="None",
            color=color,
            label=str(d),
        )

        # dashed power-law guide from fit
        lam = fit_lambda(sub)
        sfit = sub[(sub["logical_error_rate"] > 0) & (sub["logical_error_rate"] < 1)].sort_values("p_data")
        if len(sfit) >= 2 and np.isfinite(lam):
            xs = sfit["p_data"].to_numpy()[: min(4, len(sfit))]
            ys = sfit["logical_error_rate"].to_numpy()[: min(4, len(sfit))]
            a = np.exp(np.mean(np.log(ys) - lam * np.log(xs)))
            xguide = np.linspace(xs.min(), xs.max(), 100)
            yguide = a * np.power(xguide, lam)
            ax.plot(xguide, yguide, "--", color="k")

    if args.threshold is not None:
        ax.axvline(args.threshold, linestyle="--", color="red")

    ax.grid()
    ax.set_xlabel(r"$p$")
    ax.set_ylabel(r"$p_L$")
    ax.set_xscale("log")
    ax.set_yscale("log")

    left, bottom, width, height = [0.64, 0.21, 0.22, 0.22]
    ax2 = fig.add_axes([left, bottom, width, height])
    lambdas = [fit_lambda(df[df["distance"] == d]) for d in ds]
    ax2.plot(ds, lambdas, "-", c="purple")
    ax2.plot(ds, lambdas, ms=3, marker="o", markerfacecolor="None", linestyle="none", c="k")
    ax2.minorticks_on()
    ax2.set_xlabel(r"$d$", fontsize=14, labelpad=0)
    ax2.set_ylabel(r"$\lambda$", fontsize=14)
    ax2.tick_params(axis="both", which="major", labelsize=10)

    fig.legend(ds, title=r"$d$", ncol=len(ds), bbox_to_anchor=(0.12, 0.9, 0.8, 0.2))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
