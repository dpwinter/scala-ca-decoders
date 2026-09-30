from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

INPUT = Path(
    "data/pheno/"
    "pheno_tR_lifetime_Tmax120d.csv"
)

P = 0.008

D_VALUES = [
    9,
    21,
    41,
    61,
]

OUTPUT_PDF = Path(
    "figs/"
    "scala2d_pheno_reset_time.pdf"
)

OUTPUT_PNG = Path(
    "figs/"
    "scala2d_pheno_reset_time.png"
)


# =============================================================================
# Reconstruct truncated lifetimes
# =============================================================================

def summarize_lifetimes(df):
    """
    Reconstruct <T_F>_trunc from the aggregated simulation output.

    Failed trajectories contribute their actual failure time.
    Trajectories surviving to T_max contribute T_max.
    """

    rows = []

    group_cols = [
        "d",
        "p",
        "p_meas",
        "reset_factor",
        "reset_period",
        "rounds",
    ]

    for key, sub in df.groupby(
        group_cols,
        sort=True,
    ):

        (
            d,
            p,
            p_meas,
            reset_factor,
            reset_period,
            rounds,
        ) = key

        sub = sub.sort_values(
            "block"
        )

        shots_values = (
            sub["shots"]
            .astype(int)
            .unique()
        )

        if len(
            shots_values
        ) != 1:
            raise RuntimeError(
                f"Inconsistent shot count for "
                f"d={d}, p={p}, "
                f"reset_factor={reset_factor}"
            )

        shots = int(
            shots_values[0]
        )

        total_failures = int(
            sub["failures"].sum()
        )

        final_row = sub.loc[
            sub["t_end"].idxmax()
        ]

        survivors = int(
            final_row[
                "survivors_end"
            ]
        )

        if (
            total_failures
            + survivors
            != shots
        ):
            raise RuntimeError(
                f"Failure/censoring count "
                f"does not close for "
                f"d={d}, p={p}, "
                f"reset_factor="
                f"{reset_factor}: "
                f"{total_failures} failures + "
                f"{survivors} survivors "
                f"!= {shots}"
            )

        sum_failure_t = float(
            sub[
                "sum_failure_t"
            ].sum()
        )

        sum_failure_t2 = float(
            sub[
                "sum_failure_t2"
            ].sum()
        )

        t_max = float(
            rounds
        )

        sum_y = (
            sum_failure_t
            + survivors
            * t_max
        )

        sum_y2 = (
            sum_failure_t2
            + survivors
            * t_max**2
        )

        mean_tf = (
            sum_y
            / shots
        )

        if shots > 1:

            sample_var = (
                sum_y2
                - shots
                * mean_tf**2
            ) / (
                shots - 1
            )

            sample_var = max(
                sample_var,
                0.0,
            )

            se_tf = math.sqrt(
                sample_var
                / shots
            )

        else:

            se_tf = np.nan

        deficit = (
            1.0
            - mean_tf
            / t_max
        )

        deficit_se = (
            se_tf
            / t_max
        )

        rows.append(
            {
                "d":
                    int(d),

                "p":
                    float(p),

                "p_meas":
                    float(
                        p_meas
                    ),

                "reset_factor":
                    float(
                        reset_factor
                    ),

                "mean_tf":
                    mean_tf,

                "deficit":
                    deficit,

                "deficit_se":
                    deficit_se,
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Main
# =============================================================================

def main():

    df = pd.read_csv(
        INPUT
    )

    df = df[
        np.isclose(
            df["p"],
            P
        )
        & np.isclose(
            df["p_meas"],
            P
        )
        & df["d"].isin(
            D_VALUES
        )
    ].copy()

    if df.empty:

        raise RuntimeError(
            f"No data found for "
            f"p=q={P} and "
            f"d={D_VALUES}"
        )

    data = summarize_lifetimes(
        df
    )

    fig, ax = plt.subplots(
        figsize=(
            5,
            4,
        )
    )

    for d in D_VALUES:

        sub = (
            data[
                data["d"]
                == d
            ]
            .sort_values(
                "reset_factor"
            )
        )

        if sub.empty:
            continue

        ax.errorbar(
            sub[
                "reset_factor"
            ],
            sub[
                "deficit"
            ],
            yerr=sub[
                "deficit_se"
            ],
            marker="o",
            linestyle="-",
            capsize=2,
            label=rf"${d}$",
        )

    # Reference value currently used in SCALA1D.
    ax.axvline(
        0.35,
        linestyle="--",
        color="gray",
        linewidth=1.2,
    )

    ax.text(
        0.35,
        0.97,
        r"$t_R=0.35d$",
        transform=(
            ax.get_xaxis_transform()
        ),
        ha="center",
        va="top",
    )

    ax.set_xlabel(
        r"normalized reset period "
        r"$t_R/d$"
    )

    ax.set_ylabel(
        r"$1-\langle T_F\rangle_{\rm trunc}"
        r"/T_{\max}$"
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlim(
        0.10,
        0.50,
    )

    ax.grid(
        True,
        which="major",
        alpha=0.35,
    )

    ax.grid(
        False,
        which="minor",
    )

    ax.legend(
        title=r"$d$",
        loc="upper right",
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
