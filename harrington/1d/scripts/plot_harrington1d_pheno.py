# scripts/plot_harrington1d_pheno.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Input / output
# =============================================================================

INPUT = Path(
    "data/pheno/harrington1d_pheno.csv"
)

OUTPUT_DIR = Path("figs")

OUTPUT_PDF = OUTPUT_DIR / "har1d_pheno.pdf"
OUTPUT_PNG = OUTPUT_DIR / "har1d_pheno.png"


# =============================================================================
# EXACT plotting configuration from the original notebook
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
# Fit windows
# =============================================================================

FIT_WINDOWS = {
    3:  (1.0e-3, 1.0e-2),
    9:  (1.0e-3, 1.0e-2),
    27: (4.0e-3, 1.0e-2),
    81: (9.0e-3, 1.0e-2),
}


# =============================================================================
# Fit
# =============================================================================

def weighted_log_fit(sub):

    r = sub["rate"].to_numpy(
        dtype=float
    )

    T = sub["mean_TF"].to_numpy(
        dtype=float
    )

    T_se = sub["se_TF"].to_numpy(
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
        np.exp(log_A)
    )

    lam = float(
        -slope
    )

    lam_se = float(
        np.sqrt(
            covariance[1, 1]
        )
    )

    return {
        "A": A,
        "lambda": lam,
        "lambda_se": lam_se,
    }


# =============================================================================
# Load CSV
# =============================================================================

df = pd.read_csv(
    INPUT
)


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

    for d in Ls:

        lo, hi = (
            FIT_WINDOWS[d]
        )

        sub = df[
            (df["channel"] == channel)
            & (df["d"] == d)
            & (df["rate"] >= lo)
            & (df["rate"] <= hi)
        ].sort_values(
            "rate"
        )

        result = weighted_log_fit(
            sub
        )

        fits[channel][d] = (
            result
        )

        print(
            f"d={d:2d}: "
            f"lambda="
            f"{result['lambda']:.4f} "
            f"+- "
            f"{result['lambda_se']:.4f}"
        )


# =============================================================================
# p-noise data
#
# Same plotting commands as original notebook.
# =============================================================================

for i, d in enumerate(Ls):

    sub = df[
        (df["channel"] == "p")
        & (df["d"] == d)
    ].sort_values(
        "rate"
    )

    x_ = sub["rate"].to_numpy()
    y_ = sub["mean_TF"].to_numpy()
    y_err = sub["se_TF"].to_numpy()

    ax[0].loglog(
        x_,
        y_,
        "o-",
        label=Ls[i],
        color=cols[i],
        markerfacecolor="None",
    )

    ax[0].errorbar(
        x_,
        y_,
        yerr=y_err,
        linestyle="none",
        color="k",
    )


ax[0].grid()

ax[0].set_ylabel(
    r"$\langle T_F \rangle$"
)

ax[0].set_xlabel(
    r"$p$"
)


# =============================================================================
# q-noise data
#
# Same plotting commands as original notebook.
# =============================================================================

for i, d in enumerate(Ls):

    sub = df[
        (df["channel"] == "q")
        & (df["d"] == d)
    ].sort_values(
        "rate"
    )

    x_ = sub["rate"].to_numpy()
    y_ = sub["mean_TF"].to_numpy()
    y_err = sub["se_TF"].to_numpy()

    ax[1].loglog(
        x_,
        y_,
        "o-",
        label=Ls[i],
        color=cols[i],
        markerfacecolor="None",
    )

    ax[1].errorbar(
        x_,
        y_,
        yerr=y_err,
        linestyle="none",
        color="k",
    )


ax[1].grid()

ax[1].set_xlabel(
    r"$q$"
)


# =============================================================================
# EXACT original limits and subplot spacing
# =============================================================================

ax[0].set_ylim([
    1,
    1e9,
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
# Fit curves
#
# ONLY substantive change relative to original notebook:
# use fitted A and lambda instead of hand-set A and lambda.
#
# Keep exact original fit-line x coordinates.
# =============================================================================

ps_fit = np.array([
    1e-2,
    9e-3,
    8e-3,
    7e-3,
    6e-3,
    5e-3,
    4e-3,
    3e-3,
    2e-3,
    1e-3,
])


for d in Ls:

    result = fits["p"][d]

    ax[0].plot(
        ps_fit,
        result["A"]
        * np.power(
            ps_fit,
            -result["lambda"],
        ),
        "--",
        color="k",
    )


for d in Ls:

    result = fits["q"][d]

    ax[1].plot(
        ps_fit,
        result["A"]
        * np.power(
            ps_fit,
            -result["lambda"],
        ),
        "--",
        color="k",
    )


# =============================================================================
# LEFT INSET
#
# EXACT position and style from original notebook.
# Only change: fitted lambdas + error bars.
# =============================================================================

left, bottom, width, height = [
    0.32,
    0.55,
    0.14,
    0.3,
]

ax2 = fig.add_axes([
    left,
    bottom,
    width,
    height,
])


fn = lambda d: (
    d ** (
        np.log(2)
        / np.log(3)
    )
)


ax2.plot(
    Ls,
    [
        fn(d)
        for d in Ls
    ],
    "-",
    c="purple",
)


p_lambdas = [
    fits["p"][d]["lambda"]
    for d in Ls
]

p_lambdas_se = [
    fits["p"][d]["lambda_se"]
    for d in Ls
]


ax2.errorbar(
    Ls,
    p_lambdas,
    yerr=p_lambdas_se,

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
# RIGHT INSET
#
# EXACT position and style from original notebook.
# Only change: fitted lambdas + error bars.
# =============================================================================

left, bottom, width, height = [
    0.735,
    0.55,
    0.14,
    0.3,
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
        fn(d)
        for d in Ls
    ],
    "-",
    c="purple",
)


q_lambdas = [
    fits["q"][d]["lambda"]
    for d in Ls
]

q_lambdas_se = [
    fits["q"][d]["lambda_se"]
    for d in Ls
]


ax2.errorbar(
    Ls,
    q_lambdas,
    yerr=q_lambdas_se,

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
# LEGEND
#
# EXACT original notebook call.
#
# Important: use the labels from the plotted axes.
# Do not construct custom legend handles.
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
#
# EXACT original save geometry.
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
