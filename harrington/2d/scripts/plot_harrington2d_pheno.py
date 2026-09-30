# scripts/plot_harrington2d_pheno.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Input / output
# =============================================================================

INPUT = Path(
    "data/pheno/harrington2d_pheno.csv"
)

OUTPUT_DIR = Path("figs")

OUTPUT_PDF = (
    OUTPUT_DIR
    / "har2d_pheno_comb.pdf"
)

OUTPUT_PNG = (
    OUTPUT_DIR
    / "har2d_pheno_comb.png"
)


# =============================================================================
# Plot configuration
#
# Keep the original figure geometry/style.
# =============================================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
})

fig, ax = plt.subplots(
    1,
    2,
    figsize=(10, 4),
    sharey=True,
)

cols = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
]

Ls = [
    3,
    9,
    27,
    81,
]


# =============================================================================
# Fit configuration
#
# Only d=3 and d=9 are fitted.
#
# d=27 and d=81 are visibly pre-asymptotic over the available numerical
# range, so no fitted exponent is reported for them.
# =============================================================================

FIT_DISTANCES = [
    3,
    9,
]


FIT_WINDOWS = {
    3: (1.0e-3, 4.0e-3),
    9: (1.0e-3, 3.0e-3),
}


# =============================================================================
# Weighted power-law fit
# =============================================================================

def weighted_log_fit(sub):
    """
    Fit

        <T_F> = A r^{-lambda}

    using weighted least squares in logarithmic coordinates,

        log(<T_F>) = log(A) - lambda log(r).

    The Monte Carlo standard error of <T_F> is propagated as

        sigma_logT ~= sigma_T / <T_F>.

    Returns:
        A
        lambda
        lambda_se
        n_fit
        chi2_red
    """

    sub = sub[
        np.isfinite(
            sub["rate"]
        )
        & np.isfinite(
            sub["mean_TF"]
        )
        & np.isfinite(
            sub["se_TF"]
        )
        & (
            sub["rate"] > 0
        )
        & (
            sub["mean_TF"] > 0
        )
        & (
            sub["se_TF"] > 0
        )
    ].copy()


    if len(sub) < 2:

        raise RuntimeError(
            "Need at least two usable points "
            "for a power-law fit."
        )


    r = sub[
        "rate"
    ].to_numpy(
        dtype=float
    )

    T = sub[
        "mean_TF"
    ].to_numpy(
        dtype=float
    )

    T_se = sub[
        "se_TF"
    ].to_numpy(
        dtype=float
    )


    x = np.log(r)
    y = np.log(T)

    sigma_y = (
        T_se / T
    )


    X = np.column_stack([
        np.ones_like(x),
        x,
    ])


    w = (
        1.0 / sigma_y**2
    )


    normal = (
        X.T
        @ (
            w[:, None]
            * X
        )
    )


    rhs = (
        X.T
        @ (
            w * y
        )
    )


    beta = np.linalg.solve(
        normal,
        rhs,
    )


    covariance = np.linalg.inv(
        normal
    )


    log_A = beta[0]
    slope = beta[1]


    A = float(
        np.exp(
            log_A
        )
    )


    lam = float(
        -slope
    )


    lam_se = float(
        np.sqrt(
            covariance[1, 1]
        )
    )


    residual = (
        y
        - X @ beta
    ) / sigma_y


    chi2 = float(
        np.sum(
            residual**2
        )
    )


    dof = (
        len(x)
        - 2
    )


    if dof > 0:

        chi2_red = (
            chi2 / dof
        )

    else:

        chi2_red = np.nan


    return {
        "A": A,
        "lambda": lam,
        "lambda_se": lam_se,
        "n_fit": len(x),
        "chi2_red": chi2_red,
    }


# =============================================================================
# Hierarchical prediction
# =============================================================================

def lambda_theory(d):

    return (
        np.asarray(
            d,
            dtype=float,
        )
        ** (
            np.log(2.0)
            / np.log(3.0)
        )
    )


# =============================================================================
# Load data
# =============================================================================

df = pd.read_csv(
    INPUT
)


required = {
    "channel",
    "d",
    "rate",
    "shots",
    "mean_TF",
    "std_TF",
    "se_TF",
}


missing = (
    required
    - set(
        df.columns
    )
)


if missing:

    raise RuntimeError(
        f"Missing columns in "
        f"{INPUT}: "
        f"{sorted(missing)}"
    )


# =============================================================================
# Fits
# =============================================================================

fits = {
    "p": {},
    "q": {},
}


for channel in [
    "p",
    "q",
]:

    print()
    print(
        f"{channel}-noise fits"
    )

    print(
        " d    fit interval          "
        "lambda +- sigma      "
        "Nfit    chi2_red"
    )

    print(
        "--    ----------------      "
        "------------------    "
        "----    --------"
    )


    for d in FIT_DISTANCES:

        lo, hi = (
            FIT_WINDOWS[d]
        )


        sub = df[
            (
                df["channel"]
                == channel
            )
            & (
                df["d"]
                == d
            )
            & (
                df["rate"]
                >= lo
            )
            & (
                df["rate"]
                <= hi
            )
        ].sort_values(
            "rate"
        )


        result = (
            weighted_log_fit(
                sub
            )
        )


        fits[
            channel
        ][d] = result


        chi = (
            result[
                "chi2_red"
            ]
        )


        if np.isfinite(
            chi
        ):

            chi_text = (
                f"{chi:.3f}"
            )

        else:

            chi_text = "n/a"


        print(
            f"{d:2d}    "
            f"[{lo:.4g}, {hi:.4g}]       "
            f"{result['lambda']:.3f} "
            f"+- "
            f"{result['lambda_se']:.3f}      "
            f"{result['n_fit']:2d}      "
            f"{chi_text}"
        )


# =============================================================================
# p-noise panel
# =============================================================================

for i, d in enumerate(
    Ls
):

    sub = df[
        (
            df["channel"]
            == "p"
        )
        & (
            df["d"]
            == d
        )
    ].sort_values(
        "rate"
    )


    ax[0].loglog(
        sub["rate"],
        sub["mean_TF"],

        "o-",

        label=Ls[i],

        color=cols[i],

        markerfacecolor="None",
    )


    ax[0].errorbar(
        sub["rate"],
        sub["mean_TF"],

        yerr=sub["se_TF"],

        linestyle="none",

        color="k",

        linewidth=0.8,

        capsize=0,

        zorder=0,
    )


# =============================================================================
# q-noise panel
# =============================================================================

for i, d in enumerate(
    Ls
):

    sub = df[
        (
            df["channel"]
            == "q"
        )
        & (
            df["d"]
            == d
        )
    ].sort_values(
        "rate"
    )


    ax[1].loglog(
        sub["rate"],
        sub["mean_TF"],

        "o-",

        label=Ls[i],

        color=cols[i],

        markerfacecolor="None",
    )


    ax[1].errorbar(
        sub["rate"],
        sub["mean_TF"],

        yerr=sub["se_TF"],

        linestyle="none",

        color="k",

        linewidth=0.8,

        capsize=0,

        zorder=0,
    )


# =============================================================================
# Axes
# =============================================================================

ax[0].grid()
ax[1].grid()


ax[0].set_ylabel(
    r"$\langle T_F \rangle$"
)


ax[0].set_xlabel(
    r"$p$"
)


ax[1].set_xlabel(
    r"$q$"
)


ax[0].set_ylim([
    1,
    1e7,
])


ax[0].set_xlim([
    1e-3,
    1,
])


ax[1].set_xlim([
    1e-3,
    1,
])


plt.subplots_adjust(
    wspace=0.15
)


# =============================================================================
# Numerical fits
#
# Only d=3 and d=9.
# =============================================================================

for d in FIT_DISTANCES:

    lo, hi = (
        FIT_WINDOWS[d]
    )


    result = (
        fits["p"][d]
    )


    rfit = np.logspace(
        np.log10(lo),
        np.log10(hi),
        200,
    )


    ax[0].plot(
        rfit,

        result["A"]
        * rfit
        ** (
            -result["lambda"]
        ),

        "--",

        color="k",
    )


for d in FIT_DISTANCES:

    lo, hi = (
        FIT_WINDOWS[d]
    )


    result = (
        fits["q"][d]
    )


    rfit = np.logspace(
        np.log10(lo),
        np.log10(hi),
        200,
    )


    ax[1].plot(
        rfit,

        result["A"]
        * rfit
        ** (
            -result["lambda"]
        ),

        "--",

        color="k",
    )


# =============================================================================
# Left inset
#
# Prediction shown for all d.
# Numerical fit points shown only for d=3 and d=9.
# =============================================================================

left, bottom, width, height = [
    0.32,
    0.55,
    0.14,
    0.30,
]


ax2 = fig.add_axes([
    left,
    bottom,
    width,
    height,
])


ax2.plot(
    Ls,

    [
        lambda_theory(d)
        for d in Ls
    ],

    "-",

    c="purple",
)


ax2.errorbar(
    FIT_DISTANCES,

    [
        fits["p"][d]["lambda"]
        for d in FIT_DISTANCES
    ],

    yerr=[
        fits["p"][d]["lambda_se"]
        for d in FIT_DISTANCES
    ],

    ms=5,

    marker="o",

    markerfacecolor="None",

    linestyle="none",

    c="k",

    ecolor="k",

    capsize=2,
)


ax2.minorticks_on()


ax2.set_xlabel(
    r"$d$",

    fontsize=14,

    labelpad=0,
)


ax2.set_ylabel(
    r"$\lambda$",

    fontsize=14,
)


ax2.tick_params(
    axis="both",

    which="major",

    labelsize=10,
)


# =============================================================================
# Right inset
# =============================================================================

left, bottom, width, height = [
    0.735,
    0.55,
    0.14,
    0.30,
]


ax2 = fig.add_axes([
    left,
    bottom,
    width,
    height,
])


ax2.plot(
    Ls,

    [
        lambda_theory(d)
        for d in Ls
    ],

    "-",

    c="purple",
)


ax2.errorbar(
    FIT_DISTANCES,

    [
        fits["q"][d]["lambda"]
        for d in FIT_DISTANCES
    ],

    yerr=[
        fits["q"][d]["lambda_se"]
        for d in FIT_DISTANCES
    ],

    ms=5,

    marker="o",

    markerfacecolor="None",

    linestyle="none",

    c="k",

    ecolor="k",

    capsize=2,
)


ax2.minorticks_on()


ax2.set_xlabel(
    r"$d$",

    fontsize=14,

    labelpad=0,
)


ax2.set_ylabel(
    r"$\lambda$",

    fontsize=14,
)


ax2.tick_params(
    axis="both",

    which="major",

    labelsize=10,
)


# =============================================================================
# Common legend
#
# Keep same figure-level legend style/placement.
# =============================================================================

plt.figlegend(
    Ls,

    title=r"$d$",

    ncol=4,

    bbox_to_anchor=(
        -0.31,
        0.9,
        1,
        0.2,
    ),
)


# =============================================================================
# Save
# =============================================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


plt.savefig(
    OUTPUT_PDF,

    dpi=300,

    bbox_inches="tight",
)


plt.savefig(
    OUTPUT_PNG,

    dpi=300,

    bbox_inches="tight",
)


plt.show()
