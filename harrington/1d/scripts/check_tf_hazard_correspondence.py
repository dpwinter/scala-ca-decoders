from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Hazard bins
# =============================================================================

def make_hazard_bins(
    failure_times,
    n_bins=12,
):
    """
    Construct adaptive time bins containing roughly equal numbers
    of failures.

    For each interval (t0, t1], estimate the discrete-time
    constant hazard by

        h = N_fail / exposure,

    where exposure is the total number of time steps spent at risk.

    This is the MLE for a geometric first-passage process.
    """

    times = np.asarray(
        failure_times,
        dtype=np.int64,
    )

    times = times[
        times > 0
    ]

    times.sort()

    n = len(times)

    if n < 10:
        return []

    # Roughly equal event count per bin.
    target = max(
        5,
        int(
            np.ceil(
                n / n_bins
            )
        ),
    )

    boundaries = [0]

    k = target

    while k < n:

        boundary = int(
            times[
                min(
                    k - 1,
                    n - 1,
                )
            ]
        )

        if boundary > boundaries[-1]:
            boundaries.append(
                boundary
            )

        k += target

    final = int(
        times[-1]
    )

    if final > boundaries[-1]:
        boundaries.append(
            final
        )

    rows = []

    for t0, t1 in zip(
        boundaries[:-1],
        boundaries[1:],
    ):

        at_risk = (
            times > t0
        )

        n_at_risk = int(
            np.count_nonzero(
                at_risk
            )
        )

        if n_at_risk == 0:
            break

        failed = (
            (times > t0)
            & (times <= t1)
        )

        n_fail = int(
            np.count_nonzero(
                failed
            )
        )

        if n_fail == 0:
            continue

        t_alive = times[
            at_risk
        ]

        # Each surviving trajectory contributes until either
        # failure or the end of this interval.
        exposure = int(
            np.sum(
                np.minimum(
                    t_alive,
                    t1,
                )
                - t0
            )
        )

        if exposure <= 0:
            continue

        h = (
            n_fail
            / exposure
        )

        # Poisson/rare-event error estimate.
        se = (
            np.sqrt(
                n_fail
            )
            / exposure
        )

        rows.append(
            {
                "t0": t0,
                "t1": t1,
                "t": 0.5 * (
                    t0 + t1
                ),
                "n_at_risk": n_at_risk,
                "n_fail": n_fail,
                "exposure": exposure,
                "h": h,
                "se": se,
            }
        )

    return rows


# =============================================================================
# Automatic stationary-tail finder
# =============================================================================

def find_stationary_tail(
    bins,
    n_total,
    min_risk_fraction=0.15,
    min_risk_absolute=20,
    min_plateau_bins=4,
    chi2_max=2.0,
):
    """
    Find the earliest statistically constant tail.

    We first discard bins whose risk set has become too depleted.
    Then, for every possible starting bin, test whether all remaining
    usable bins are compatible with one constant hazard.

    Returns
    -------
    usable_bins, plateau_index, reduced_chi2
    """

    min_risk = max(
        min_risk_absolute,
        int(
            np.ceil(
                min_risk_fraction
                * n_total
            )
        ),
    )

    usable = [
        row
        for row in bins
        if (
            row["n_at_risk"]
            >= min_risk
            and row["n_fail"] > 0
            and np.isfinite(
                row["h"]
            )
            and np.isfinite(
                row["se"]
            )
            and row["se"] > 0
        )
    ]

    if len(
        usable
    ) < min_plateau_bins:
        return (
            usable,
            None,
            np.nan,
        )

    for i in range(
        len(usable)
        - min_plateau_bins
        + 1
    ):

        tail = usable[
            i:
        ]

        h = np.asarray(
            [
                row["h"]
                for row in tail
            ]
        )

        se = np.asarray(
            [
                row["se"]
                for row in tail
            ]
        )

        weights = (
            1.0
            / se**2
        )

        h_const = (
            np.sum(
                weights * h
            )
            / np.sum(
                weights
            )
        )

        chi2 = np.sum(
            (
                (
                    h - h_const
                )
                / se
            )**2
        )

        dof = (
            len(tail) - 1
        )

        if dof <= 0:
            continue

        reduced_chi2 = (
            chi2 / dof
        )

        if reduced_chi2 <= chi2_max:

            return (
                usable,
                i,
                reduced_chi2,
            )

    return (
        usable,
        None,
        np.nan,
    )


# =============================================================================
# Plateau-rate estimator
# =============================================================================

def pooled_tail_hazard(
    usable_bins,
    plateau_index,
):
    """
    Pool all plateau bins:

        h_inf = total failures / total exposure.

    This is preferable to averaging the individual bin estimates.
    """

    tail = usable_bins[
        plateau_index:
    ]

    failures = sum(
        row["n_fail"]
        for row in tail
    )

    exposure = sum(
        row["exposure"]
        for row in tail
    )

    if (
        failures <= 0
        or exposure <= 0
    ):
        return (
            np.nan,
            np.nan,
        )

    h = (
        failures
        / exposure
    )

    se = (
        np.sqrt(
            failures
        )
        / exposure
    )

    return h, se


# =============================================================================
# Bootstrap R = h_inf * <T_F>
# =============================================================================

def bootstrap_ratio(
    failure_times,
    t0,
    t1,
    n_bootstrap,
    rng,
):
    """
    Bootstrap the quantity

        R = h_inf * <T_F>

    using the plateau interval [t0, t1] determined from the
    original sample.

    The plateau hazard is re-estimated from failures/exposure
    inside that same time window for each bootstrap sample.
    """

    times = np.asarray(
        failure_times,
        dtype=np.int64,
    )

    n = len(times)

    ratios = []

    for _ in range(
        n_bootstrap
    ):

        sample = rng.choice(
            times,
            size=n,
            replace=True,
        )

        mean_tf = float(
            np.mean(
                sample
            )
        )

        at_risk = (
            sample > t0
        )

        if not np.any(
            at_risk
        ):
            continue

        s = sample[
            at_risk
        ]

        failed = (
            (s > t0)
            & (s <= t1)
        )

        n_fail = int(
            np.count_nonzero(
                failed
            )
        )

        exposure = int(
            np.sum(
                np.minimum(
                    s,
                    t1,
                )
                - t0
            )
        )

        if (
            n_fail <= 0
            or exposure <= 0
        ):
            continue

        h = (
            n_fail
            / exposure
        )

        ratios.append(
            h * mean_tf
        )

    if len(
        ratios
    ) < 20:

        return (
            np.nan,
            np.nan,
        )

    ratios = np.asarray(
        ratios
    )

    return tuple(
        np.percentile(
            ratios,
            [
                2.5,
                97.5,
            ],
        )
    )


# =============================================================================
# One (d,p,q) dataset
# =============================================================================

def analyse_group(
    group,
    n_bins,
    min_risk_fraction,
    min_risk_absolute,
    min_plateau_bins,
    chi2_max,
    n_bootstrap,
    rng,
):
    times = np.asarray(
        group["T_F"],
        dtype=np.int64,
    )

    n = len(times)

    mean_tf = float(
        np.mean(
            times
        )
    )

    se_mean_tf = float(
        np.std(
            times,
            ddof=1,
        )
        / np.sqrt(n)
    )

    inv_mean = (
        1.0
        / mean_tf
    )

    inv_mean_se = (
        se_mean_tf
        / mean_tf**2
    )

    bins = make_hazard_bins(
        times,
        n_bins=n_bins,
    )

    (
        usable,
        plateau_index,
        reduced_chi2,
    ) = find_stationary_tail(
        bins,
        n_total=n,
        min_risk_fraction=(
            min_risk_fraction
        ),
        min_risk_absolute=(
            min_risk_absolute
        ),
        min_plateau_bins=(
            min_plateau_bins
        ),
        chi2_max=chi2_max,
    )

    if plateau_index is None:

        return {
            "N": n,
            "mean_TF": mean_tf,
            "SE_mean_TF": se_mean_tf,
            "inv_mean_TF": inv_mean,
            "SE_inv_mean_TF": (
                inv_mean_se
            ),
            "h_inf": np.nan,
            "SE_h_inf": np.nan,
            "R": np.nan,
            "R_lo": np.nan,
            "R_hi": np.nan,
            "plateau_start": np.nan,
            "plateau_end": np.nan,
            "plateau_bins": 0,
            "chi2_red": np.nan,
            "stationary": False,
            "hazard_bins": bins,
            "usable_bins": usable,
            "plateau_index": None,
        }

    h_inf, se_h_inf = (
        pooled_tail_hazard(
            usable,
            plateau_index,
        )
    )

    tail = usable[
        plateau_index:
    ]

    plateau_start = int(
        tail[0]["t0"]
    )

    plateau_end = int(
        tail[-1]["t1"]
    )

    ratio = (
        h_inf
        * mean_tf
    )

    if n_bootstrap > 0:

        ratio_lo, ratio_hi = (
            bootstrap_ratio(
                times,
                plateau_start,
                plateau_end,
                n_bootstrap,
                rng,
            )
        )

    else:

        ratio_lo = np.nan
        ratio_hi = np.nan

    return {
        "N": n,
        "mean_TF": mean_tf,
        "SE_mean_TF": se_mean_tf,
        "inv_mean_TF": inv_mean,
        "SE_inv_mean_TF": (
            inv_mean_se
        ),
        "h_inf": h_inf,
        "SE_h_inf": se_h_inf,
        "R": ratio,
        "R_lo": ratio_lo,
        "R_hi": ratio_hi,
        "plateau_start": (
            plateau_start
        ),
        "plateau_end": (
            plateau_end
        ),
        "plateau_bins": len(
            tail
        ),
        "chi2_red": (
            reduced_chi2
        ),
        "stationary": True,
        "hazard_bins": bins,
        "usable_bins": usable,
        "plateau_index": (
            plateau_index
        ),
    }


# =============================================================================
# Diagnostic hazard plot
# =============================================================================

def plot_group_hazard(
    result,
    title,
    output,
):
    bins = result[
        "hazard_bins"
    ]

    if not bins:
        return

    t = np.asarray(
        [
            row["t"]
            for row in bins
        ]
    )

    h = np.asarray(
        [
            row["h"]
            for row in bins
        ]
    )

    se = np.asarray(
        [
            row["se"]
            for row in bins
        ]
    )

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    ax.errorbar(
        t,
        h,
        yerr=se,
        marker="o",
        markerfacecolor="none",
        linestyle="-",
        capsize=2,
    )

    if result[
        "stationary"
    ]:

        ax.axvline(
            result[
                "plateau_start"
            ],
            linestyle=":",
        )

        ax.axhline(
            result[
                "h_inf"
            ],
            linestyle="--",
            label=(
                rf"$h_\infty="
                rf"{result['h_inf']:.2e}$"
            ),
        )

        ax.axhline(
            result[
                "inv_mean_TF"
            ],
            linestyle="-.",
            label=(
                rf"$1/\langle T_F\rangle="
                rf"{result['inv_mean_TF']:.2e}$"
            ),
        )

        ax.legend()

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        r"time $t$"
    )

    ax.set_ylabel(
        r"hazard $h(t)$"
    )

    ax.set_title(
        title
    )

    ax.grid(
        True,
        which="major",
    )

    ax.grid(
        False,
        which="minor",
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# Global comparison plot
# =============================================================================

def plot_correspondence(
    summary,
    output,
):
    df = summary[
        summary[
            "stationary"
        ]
    ].copy()

    if len(df) == 0:
        return

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    for mode, group in df.groupby(
        "noise_mode"
    ):

        ax.errorbar(
            group[
                "inv_mean_TF"
            ],
            group[
                "h_inf"
            ],
            xerr=group[
                "SE_inv_mean_TF"
            ],
            yerr=group[
                "SE_h_inf"
            ],
            marker="o",
            markerfacecolor="none",
            linestyle="none",
            capsize=2,
            label=mode,
        )

    values = np.concatenate(
        [
            df[
                "inv_mean_TF"
            ].to_numpy(),
            df[
                "h_inf"
            ].to_numpy(),
        ]
    )

    values = values[
        np.isfinite(
            values
        )
        & (
            values > 0
        )
    ]

    lo = values.min()
    hi = values.max()

    ax.plot(
        [
            lo,
            hi,
        ],
        [
            lo,
            hi,
        ],
        linestyle="--",
        label=r"$h_\infty=1/\langle T_F\rangle$",
    )

    ax.set_xscale(
        "log"
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        r"$1/\langle T_F\rangle$"
    )

    ax.set_ylabel(
        r"$h_\infty$"
    )

    ax.grid(
        True,
        which="major",
    )

    ax.grid(
        False,
        which="minor",
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def plot_ratio(
    summary,
    output,
):
    df = summary[
        summary[
            "stationary"
        ]
    ].copy()

    if len(df) == 0:
        return

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    for mode, group in df.groupby(
        "noise_mode"
    ):

        ax.errorbar(
            group[
                "inv_mean_TF"
            ],
            group[
                "R"
            ],
            yerr=[
                (
                    group["R"]
                    - group["R_lo"]
                ).to_numpy(),
                (
                    group["R_hi"]
                    - group["R"]
                ).to_numpy(),
            ],
            marker="o",
            markerfacecolor="none",
            linestyle="none",
            capsize=2,
            label=mode,
        )

    ax.axhline(
        1.0,
        linestyle="--",
    )

    ax.set_xscale(
        "log"
    )

    ax.set_xlabel(
        r"$1/\langle T_F\rangle$"
    )

    ax.set_ylabel(
        r"$h_\infty\langle T_F\rangle$"
    )

    ax.grid(
        True,
        which="major",
    )

    ax.grid(
        False,
        which="minor",
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# Main
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/pheno/"
            "harrington1d_pheno_TF_raw_combined.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/pheno/"
            "harrington1d_tf_hazard_summary.csv"
        ),
    )

    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path(
            "figs/tf_hazard_check"
        ),
    )

    parser.add_argument(
        "--n-bins",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--min-risk-fraction",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--min-risk-absolute",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--min-plateau-bins",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--chi2-max",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    df = pd.read_csv(
        args.input
    )

    required = {
        "noise_mode",
        "d",
        "p",
        "q",
        "U",
        "sample_id",
        "T_F",
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    args.figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(
        args.seed
    )

    rows = []

    group_columns = [
        "noise_mode",
        "d",
        "p",
        "q",
        "U",
    ]

    for key, group in df.groupby(
        group_columns,
        sort=True,
    ):

        (
            noise_mode,
            d,
            p,
            q,
            U,
        ) = key

        result = analyse_group(
            group,
            n_bins=args.n_bins,
            min_risk_fraction=(
                args.min_risk_fraction
            ),
            min_risk_absolute=(
                args.min_risk_absolute
            ),
            min_plateau_bins=(
                args.min_plateau_bins
            ),
            chi2_max=args.chi2_max,
            n_bootstrap=args.bootstrap,
            rng=rng,
        )

        row = {
            "noise_mode": noise_mode,
            "d": int(d),
            "p": float(p),
            "q": float(q),
            "U": int(U),
        }

        for name, value in (
            result.items()
        ):

            if name in {
                "hazard_bins",
                "usable_bins",
                "plateau_index",
            }:
                continue

            row[name] = value

        rows.append(
            row
        )

        print(
            f"{noise_mode:12s} "
            f"d={int(d):3d} "
            f"p={p:.4g} "
            f"q={q:.4g} "
            f"N={result['N']:5d} "
            f"<TF>={result['mean_TF']:.4e} "
            f"1/<TF>={result['inv_mean_TF']:.4e} "
            f"h_inf={result['h_inf']:.4e} "
            f"R={result['R']:.4f} "
            f"chi2={result['chi2_red']:.3f} "
            f"{'OK' if result['stationary'] else 'NO PLATEAU'}"
        )

        title = (
            f"{noise_mode}, "
            f"d={int(d)}, "
            f"p={p:g}, q={q:g}"
        )

        filename = (
            f"{noise_mode}_"
            f"d{int(d)}_"
            f"p{p:.4g}_"
            f"q{q:.4g}.png"
        )

        plot_group_hazard(
            result,
            title,
            args.figure_dir
            / filename,
        )

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        args.output,
        index=False,
    )

    plot_correspondence(
        summary,
        args.figure_dir
        / "tf_vs_stationary_hazard.png",
    )

    plot_ratio(
        summary,
        args.figure_dir
        / "hazard_times_mean_tf.png",
    )

    print()
    print("=" * 100)
    print(
        f"Saved summary: {args.output}"
    )

    print(
        "Saved diagnostics: "
        f"{args.figure_dir}"
    )

    good = summary[
        summary["stationary"]
    ]

    if len(good):

        ratio = good[
            "R"
        ].to_numpy()

        print()
        print(
            "Stationary groups:"
        )

        print(
            f"    N groups              = "
            f"{len(good)}"
        )

        print(
            f"    median h_inf<TF>      = "
            f"{np.nanmedian(ratio):.4f}"
        )

        print(
            f"    min h_inf<TF>         = "
            f"{np.nanmin(ratio):.4f}"
        )

        print(
            f"    max h_inf<TF>         = "
            f"{np.nanmax(ratio):.4f}"
        )

        print(
            f"    median |R-1|          = "
            f"{np.nanmedian(np.abs(ratio - 1.0)):.4f}"
        )

    print("=" * 100)


if __name__ == "__main__":
    main()
