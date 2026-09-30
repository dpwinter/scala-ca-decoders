# scripts/plot_code_capacity.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA = Path("data/code_capacity/code_capacity.csv")
OUT = Path("figs")

Ls = [3, 9, 27, 81]
cols = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
})


def get_column(df, *candidates):
    for name in candidates:
        if name in df.columns:
            return name

    raise KeyError(
        f"Could not find any of {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def majority_vote_3(p):
    return 3.0 * p**2 - 2.0 * p**3


def concatenated_majority_vote(p, d):
    q = np.asarray(p, dtype=float).copy()

    while d > 1:
        if d % 3 != 0:
            raise ValueError(f"d={d} is not a power of 3")

        q = majority_vote_3(q)
        d //= 3

    return q


def main():
    df = pd.read_csv(DATA)

    d_col = get_column(df, "d", "distance")
    p_col = get_column(df, "p", "p_data", "physical_error_rate")
    pl_col = get_column(
        df,
        "p_L",
        "pL",
        "p_l",
        "logical_error_rate",
        "logical_error_probability",
    )

    # EXACT plotting geometry from the notebook.
    fig = plt.figure(figsize=(5, 4))

    # Colored numerical curves.
    for i, d in enumerate(Ls):
        sub = df[df[d_col] == d].sort_values(p_col)

        plt.plot(
            sub[p_col],
            sub[pl_col],
            linestyle="-",
            marker="o",
            markerfacecolor="None",
            c=cols[i],
            zorder=2,
        )

    # Black concatenated-majority curves.
    # Plot afterwards so they sit above the colored curves.
    for d in Ls:
        sub = df[df[d_col] == d].sort_values(p_col)

        p = sub[p_col].to_numpy()

        plt.plot(
            p,
            concatenated_majority_vote(p, d),
            "--",
            color="k",
            zorder=3,
        )

    # EXACT legend logic from the notebook.
    plt.legend(
        Ls,
        title=r"$d$",
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0,
        ncol=4,
    )

    plt.yscale("log")
    plt.xscale("log")

    plt.ylabel(r"$p_L$")
    plt.xlabel(r"$p$")

    # Your original reference plot has both x- and y-major grid lines.
    plt.grid()

    # Majority-voting threshold.
    plt.axvline(
        0.5,
        linestyle="--",
        color="red",
        zorder=4,
    )

    plt.xlim([1e-2, 1.0])
    plt.ylim([1e-6, 1.0])

    OUT.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        OUT / "code_capacity.pdf",
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        OUT / "code_capacity.png",
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Wrote {OUT / 'code_capacity.pdf'}")
    print(f"Wrote {OUT / 'code_capacity.png'}")


if __name__ == "__main__":
    main()
