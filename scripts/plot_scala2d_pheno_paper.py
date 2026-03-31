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


def add_stats_from_datafile(df: pd.DataFrame) -> pd.DataFrame:
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
    df = df[df["reset_time"] <= df["distance"]].copy()
    idx = df.groupby(["distance", "p_data"])["mean_from_file"].idxmax()
    return df.loc[idx].copy().sort_values(["distance", "p_data"])


def fit_lambda(sub: pd.DataFrame, nfit: int = 5) -> float:
    s = sub.sort_values("p_data")
    s = s[s["mean_from_file"] > 0]
    if len(s) < 2:
        return np.nan
    s = s.iloc[: min(nfit, len(s))]
    x = np.log(s["p_data"].to_numpy())
    y = np.log(s["mean_from_file"].to_numpy())
    m, _ = np.polyfit(x, y, 1)
    return -float(m)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Summary CSV for SCALA2D phenomenological scan")
    parser.add_argument("--output", required=True, help="Output image path")
    args = parser.parse_args()

    df = load_summary(Path(args.csv))
    df = df[df["model"] == "scala2d"].copy()

    # If your intended paper plot is p=q=psig, keep only that manifold.
    if {"p_meas", "p_signal"}.issubset(df.columns):
        df = df[
            np.isclose(df["p_data"], df["p_meas"]) &
            np.isclose(df["p_data"], df["p_signal"])
        ].copy()

    df = add_stats_from_datafile(df)
    df = select_best_reset(df)

    ds = sorted(df["distance"].unique())

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for i, d in enumerate(ds):
        sub = df[df["distance"] == d].sort_values("p_data")
        color = COLORS[i % len(COLORS)]
        ax.plot(
            sub["p_data"],
            sub["mean_from_file"],
            linestyle="-",
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
    ax.set_xlabel(r"$p = q = p_{\mathrm{sig}}$")
    ax.set_ylabel(r"$\langle T_F \rangle$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([1e-3, 1])

    left, bottom, width, height = [0.64, 0.56, 0.22, 0.22]
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
