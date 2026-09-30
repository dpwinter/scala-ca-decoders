from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TIMESERIES = Path(
    "data/pheno/"
    "pheno_tR_firstpassage_timeseries.csv"
)

SUMMARY = Path(
    "data/pheno/"
    "pheno_tR_firstpassage_summary.csv"
)

OUTPUT_SUMMARY = Path(
    "figs/"
    "scala2d_pheno_tR_firstpassage.pdf"
)

OUTPUT_DIAGNOSTIC = Path(
    "figs/"
    "scala2d_pheno_tR_firstpassage_diagnostic.pdf"
)


def main():

    ts = pd.read_csv(
        TIMESERIES
    )

    summary = pd.read_csv(
        SUMMARY
    )

    # =========================================================================
    # Figure 1:
    # stationary hazard versus normalized reset period
    # =========================================================================

    fig, ax = plt.subplots(
        figsize=(
            5.0,
            4.0,
        )
    )

    for d in sorted(
        summary["d"].unique()
    ):

        sub = (
            summary[
                (summary["d"] == d)
                & (
                    summary[
                        "plateau_ok"
                    ]
                    == True
                )
            ]
            .sort_values(
                "actual_reset_factor"
            )
            .copy()
        )

        if sub.empty:
            continue

        ax.errorbar(
            sub[
                "actual_reset_factor"
            ],
            sub[
                "h_inf"
            ],
            yerr=sub[
                "SE_h_inf"
            ],

            marker="o",
            linestyle="-",

            markerfacecolor="none",
            markersize=5,

            linewidth=1.4,
            capsize=2,

            label=rf"${int(d)}$",
        )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        r"normalized reset period "
        r"$t_R/d$"
    )

    ax.set_ylabel(
        r"stationary logical hazard "
        r"$h_\infty$"
    )

    ax.grid(
        True,
        which="major",
        alpha=0.35,
    )

    ax.grid(
        True,
        which="minor",
        alpha=0.10,
    )

    ax.legend(
        title=r"$d$",
        loc="best",
    )

    fig.tight_layout()

    OUTPUT_SUMMARY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        OUTPUT_SUMMARY,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_SUMMARY.with_suffix(
            ".png"
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # =========================================================================
    # Figure 2:
    # full h(t) curves + detected plateau intervals
    # =========================================================================

    distances = sorted(
        ts["d"].unique()
    )

    fig, axes = plt.subplots(
        len(distances),
        1,
        figsize=(
            7.0,
            3.0
            * len(distances),
        ),
        squeeze=False,
    )

    axes = axes[:, 0]

    for ax, d in zip(
        axes,
        distances,
    ):

        dsub = ts[
            ts["d"] == d
        ].copy()

        periods = sorted(
            dsub[
                "requested_reset_period"
            ].unique()
        )

        for tR in periods:

            sub = (
                dsub[
                    dsub[
                        "requested_reset_period"
                    ]
                    == tR
                ]
                .sort_values(
                    "t_start"
                )
                .copy()
            )

            good = (
                (sub["failures"] > 0)
                & (sub["exposure"] > 0)
            )

            sub = sub[
                good
            ]

            if sub.empty:
                continue

            t = (
                0.5
                * (
                    sub[
                        "t_start"
                    ].to_numpy(
                        dtype=float
                    )
                    + sub[
                        "t_end"
                    ].to_numpy(
                        dtype=float
                    )
                )
            )

            h = (
                sub[
                    "failures"
                ].to_numpy(
                    dtype=float
                )
                / sub[
                    "exposure"
                ].to_numpy(
                    dtype=float
                )
            )

            se = (
                np.sqrt(
                    sub[
                        "failures"
                    ].to_numpy(
                        dtype=float
                    )
                )
                / sub[
                    "exposure"
                ].to_numpy(
                    dtype=float
                )
            )

            ax.errorbar(
                t,
                h,
                yerr=se,

                marker="o",
                linestyle="-",

                markerfacecolor="none",
                markersize=3.5,

                linewidth=1.0,
                capsize=1.5,

                label=(
                    rf"$t_R={int(tR)}$"
                ),
            )

            # -------------------------------------------------------------
            # Overlay detected plateau as horizontal segment
            # -------------------------------------------------------------

            srow = summary[
                (summary["d"] == d)
                & (
                    summary[
                        "reset_period"
                    ]
                    == tR
                )
                & (
                    summary[
                        "plateau_ok"
                    ]
                    == True
                )
            ]

            if not srow.empty:

                t0 = float(
                    srow[
                        "plateau_start"
                    ].iloc[0]
                )

                t1 = float(
                    srow[
                        "plateau_end"
                    ].iloc[0]
                )

                h_inf = float(
                    srow[
                        "h_inf"
                    ].iloc[0]
                )

                ax.hlines(
                    h_inf,
                    t0,
                    t1,
                    linewidth=2.2,
                )

        ax.set_yscale(
            "log"
        )

        ax.set_ylabel(
            r"$h(t)$"
        )

        ax.set_title(
            rf"$d={int(d)}$"
        )

        ax.grid(
            True,
            which="major",
            alpha=0.35,
        )

        ax.grid(
            True,
            which="minor",
            alpha=0.10,
        )

        ax.legend(
            ncol=2,
            fontsize=8,
        )

    axes[-1].set_xlabel(
        r"time $t$"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIAGNOSTIC,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_DIAGNOSTIC.with_suffix(
            ".png"
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {OUTPUT_SUMMARY}"
    )

    print(
        f"Saved: "
        f"{OUTPUT_SUMMARY.with_suffix('.png')}"
    )

    print(
        f"Saved: {OUTPUT_DIAGNOSTIC}"
    )

    print(
        f"Saved: "
        f"{OUTPUT_DIAGNOSTIC.with_suffix('.png')}"
    )


if __name__ == "__main__":
    main()
