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


COLORS = [
    "tab:blue",
    "lightsteelblue",
    "burlywood",
    "tab:orange",
    "gold",
    "yellowgreen",
    "olivedrab",
    "tab:red",
]


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def add_stats_from_datafile(df: pd.DataFrame) -> pd.DataFrame:
    means = []
    errs = []
    ns = []

    for _, row in df.iterrows():
        y = np.load(row["data_file"])
        means.append(float(np.mean(y)))
        errs.append(float(np.std(y, ddof=0) / np.sqrt(len(y))))
        ns.append(int(len(y)))

    out = df.copy()
    out["mean_from_file"] = means
    out["stderr_from_file"] = errs
    out["n_samples_file"] = ns
    return out


def select_best_reset(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    # Only allow reset times <= distance
    df = df[df["reset_time"] <= df["distance"]].copy()

    idx = df.groupby(["distance", value_col])["mean_from_file"].idxmax()
    return df.loc[idx].copy().sort_values(["distance", value_col])


def fit_lambda(sub: pd.DataFrame, xcol: str, nfit: int = 6) -> float:
    s = sub.sort_values(xcol).dropna()
    s = s[s["mean_from_file"] > 0]
    if len(s) < 2:
        return np.nan
    s = s.iloc[: min(nfit, len(s))]
    x = np.log(s[xcol].to_numpy())
    y = np.log(s["mean_from_file"].to_numpy())
    m, _ = np.polyfit(x, y, 1)
    return -float(m)


def lower_bound_scala1d_p(d: int) -> float:
    return (1.0 + np.sqrt(d)) ** 2 / 4.0


def exact_scala1d_q(d: int) -> float:
    return (d + 3.0) / 2.0


def plot_panel(ax, df: pd.DataFrame, xcol: str, xlabel: str, lambdas_exact_fn, inset_rect):
    ds = sorted(df["distance"].unique())

    for i, d in enumerate(ds):
        sub = df[df["distance"] == d].sort_values(xcol)
        color = COLORS[i % len(COLORS)]
        ax.plot(
            sub[xcol],
            sub["mean_from_file"],
            linestyle="-",
            marker="o",
            markerfacecolor="None",
            color=color,
            label=str(d),
        )
        ax.errorbar(
            sub[xcol],
            sub["mean_from_file"],
            yerr=sub["stderr_from_file"],
            c="k",
            linestyle="none",
        )

    ax.grid()
    ax.set_xlabel(xlabel)
    ax.set_xscale("log")

    lambdas_fit = [fit_lambda(df[df["distance"] == d], xcol) for d in ds]

    left, bottom, width, height = inset_rect
    ax2 = ax.figure.add_axes([left, bottom, width, height])
    ax2.plot(ds, [lambdas_exact_fn(d) for d in ds], "-", c="purple")
    ax2.plot(ds, lambdas_fit, ms=3, marker="o", markerfacecolor="None", linestyle="none", c="k")
    ax2.minorticks_on()
    ax2.set_xlabel(r"$d$", fontsize=14, labelpad=0)
    ax2.set_ylabel(r"$\lambda$", fontsize=14)
    ax2.tick_params(axis="both", which="major", labelsize=10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p-csv", required=True, help="Summary CSV for pure data-noise scan")
    parser.add_argument("--q-csv", required=True, help="Summary CSV for pure measurement-noise scan")
    parser.add_argument("--output", required=True, help="Output image path")
    args = parser.parse_args()

    p_df = add_stats_from_datafile(load_summary(Path(args.p_csv)))
    q_df = add_stats_from_datafile(load_summary(Path(args.q_csv)))

    # pure data-noise panel
    p_df = p_df[(p_df["p_meas"] == 0.0) & (p_df["p_signal"] == 0.0)].copy()
    p_df = select_best_reset(p_df, "p_data")

    # pure measurement-noise panel
    q_df = q_df[(q_df["p_data"] == 0.0) & (q_df["p_signal"] == 0.0)].copy()
    q_df = select_best_reset(q_df, "p_meas")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    plot_panel(
        ax[0],
        p_df,
        xcol="p_data",
        xlabel=r"$p$",
        lambdas_exact_fn=lower_bound_scala1d_p,
        inset_rect=[0.37, 0.55, 0.1, 0.3],
    )
    plot_panel(
        ax[1],
        q_df,
        xcol="p_meas",
        xlabel=r"$q$",
        lambdas_exact_fn=exact_scala1d_q,
        inset_rect=[0.785, 0.55, 0.1, 0.3],
    )

    ax[0].set_ylabel(r"$\langle T_F \rangle$")
    ax[0].set_yscale("log")
    ax[0].set_ylim([1, 1e8])
    ax[0].set_xlim([1e-3, 1])
    ax[1].set_xlim([1e-3, 1])

    ds = sorted(p_df["distance"].unique())
    fig.legend(ds, title=r"$d$", ncol=len(ds), bbox_to_anchor=(-0.182, 0.9, 1, 0.2))
    plt.subplots_adjust(wspace=0.15)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
