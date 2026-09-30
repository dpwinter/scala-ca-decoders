# scripts/plot_scala_screening.py

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FILES = [
    Path("data/pheno/scala_screening_d301.csv"),
    Path("data/pheno/scala_screening_d401.csv"),
    Path("data/pheno/scala_screening_d501.csv"),
]

OUTPUT_PDF = Path("figs/scala_screening.pdf")
OUTPUT_PNG = Path("figs/scala_screening.png")


def main():

    fig, ax = plt.subplots(figsize=(4.8, 3.4))

    markers = ["o", "s", "^"]

    for path, marker in zip(FILES, markers):

        df = pd.read_csv(path).sort_values("r0")

        d = int(df["d"].iloc[0])

        ax.errorbar(
            df["r0"],
            df["mean_dR"],
            yerr=df["se_dR"],
            marker=marker,
            markersize=5,
            linewidth=1.5,
            capsize=2,
            label=rf"${d}$",
        )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
        color="0.5",
    )

    ax.set_xlabel(r"Initial separation $r_0$")
    ax.set_ylabel(r"$\langle \Delta R\rangle_{\rm surv}$")

    ax.legend(
        title=r"$d$",
        loc="lower right",
    )

    ax.text(
        0.50,
        0.05,
        r"$p=q=0.02$",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
    )

    OUTPUT_PDF.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()
