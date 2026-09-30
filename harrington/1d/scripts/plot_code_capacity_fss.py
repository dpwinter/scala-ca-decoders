"""
Harrington1D code-capacity finite-size scaling.

Fit:
    p_L = F(x)
    x = (p - p_c) d^(1/nu)

with a cubic approximation to F.

Run:
    python -m scripts.plot_code_capacity_fss
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


DATA_FILE = Path(
    "data/code_capacity/code_capacity_fss.csv"
)

FIG_DIR = Path("figs")
FIG_PDF = FIG_DIR / "code_capacity_fss.pdf"
FIG_PNG = FIG_DIR / "code_capacity_fss.png"


# ============================================================
# Plot style
# ============================================================

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 14,
        "legend.fontsize": 11,
    }
)


# ============================================================
# Fit settings
# ============================================================

D_MIN = 3

P_MIN = 0.44
P_MAX = 0.56

PC_EXACT = 0.5
NU_EXACT = np.log(3) / np.log(1.5)


# ============================================================
# Scaling model
# ============================================================

def scaling_x(p, d, pc, nu):
    return (
        (p - pc)
        * d ** (1.0 / nu)
    )


def model(params, p, d):
    pc, nu, a0, a1, a2, a3 = params

    x = scaling_x(
        p,
        d,
        pc,
        nu,
    )

    return (
        a0
        + a1 * x
        + a2 * x**2
        + a3 * x**3
    )


def residuals(
    params,
    p,
    d,
    pL,
    se,
):
    return (
        model(params, p, d)
        - pL
    ) / se


# ============================================================
# Load data
# ============================================================

data = pd.read_csv(
    DATA_FILE
)

data = data[
    (data["d"] >= D_MIN)
    & (data["p"] >= P_MIN)
    & (data["p"] <= P_MAX)
    & np.isfinite(data["pL"])
    & np.isfinite(data["se"])
    & (data["se"] > 0)
].copy()


p = data["p"].to_numpy(
    dtype=float
)

d = data["d"].to_numpy(
    dtype=float
)

pL = data["pL"].to_numpy(
    dtype=float
)

se = data["se"].to_numpy(
    dtype=float
)


# ============================================================
# Fit
# ============================================================

initial = [
    0.5,
    2.7,
    0.5,
    0.5,
    0.0,
    0.0,
]

fit = least_squares(
    residuals,
    initial,
    args=(
        p,
        d,
        pL,
        se,
    ),
    bounds=(
        [
            0.47,
            1.0,
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
        ],
        [
            0.53,
            5.0,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
        ],
    ),
)

pc, nu, a0, a1, a2, a3 = fit.x


# ============================================================
# Fit uncertainty
# ============================================================

dof = len(pL) - len(fit.x)

chi2 = np.sum(
    fit.fun**2
)

chi2_red = chi2 / dof

cov = np.linalg.pinv(
    fit.jac.T @ fit.jac
)

cov *= chi2_red

errors = np.sqrt(
    np.diag(cov)
)

pc_err = errors[0]
nu_err = errors[1]


# ============================================================
# Report
# ============================================================

print()
print("=" * 60)
print("Harrington1D code-capacity FSS")
print("=" * 60)

print(f"d >=       {D_MIN}")
print(f"p window   [{P_MIN}, {P_MAX}]")
print(f"N points   {len(data)}")
print()

print(
    f"p_c        = "
    f"{pc:.8f} +/- {pc_err:.8f}"
)

print(
    f"nu         = "
    f"{nu:.6f} +/- {nu_err:.6f}"
)

print(
    f"chi2       = "
    f"{chi2:.3f}"
)

print(
    f"dof        = "
    f"{dof}"
)

print(
    f"chi2_red   = "
    f"{chi2_red:.3f}"
)

print()
print("Exact concatenated-majority values:")
print(f"p_c exact  = {PC_EXACT:.8f}")
print(f"nu exact   = {NU_EXACT:.6f}")


# ============================================================
# Collapse
# ============================================================

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# Match the geometry of the code-capacity plotting style.
fig = plt.figure(
    figsize=(5, 4)
)

ax = plt.gca()

distances = sorted(
    data["d"].unique()
)

all_x = []
all_y = []


for distance in distances:

    subset = data[
        data["d"] == distance
    ].sort_values("p")

    x = scaling_x(
        subset["p"].to_numpy(float),
        float(distance),
        pc,
        nu,
    )

    y = subset["pL"].to_numpy(
        float
    )

    yerr = subset["se"].to_numpy(
        float
    )

    all_x.extend(x)
    all_y.extend(y)

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        linestyle="-",
        marker="o",
        markerfacecolor="none",
        capsize=2,
        label=rf"${int(distance)}$",
        zorder=2,
    )


# ============================================================
# Fitted scaling function
# ============================================================

x_fit = np.linspace(
    min(all_x),
    max(all_x),
    500,
)

y_fit = (
    a0
    + a1 * x_fit
    + a2 * x_fit**2
    + a3 * x_fit**3
)

ax.plot(
    x_fit,
    y_fit,
    "--",
    color="black",
    zorder=5,
)


# ============================================================
# Style
# ============================================================

ax.set_xlabel(
    r"$(p-p_c)d^{1/\nu}$"
)

ax.set_ylabel(
    r"$p_L$"
)

ax.grid()


# ============================================================
# Tight plotting limits
# ============================================================

x_min = min(all_x)
x_max = max(all_x)

y_min = min(all_y)
y_max = max(all_y)

x_range = x_max - x_min
y_range = y_max - y_min

x_pad = 0.04 * x_range
y_pad = 0.08 * y_range

ax.set_xlim(
    x_min - x_pad,
    x_max + x_pad,
)

ax.set_ylim(
    y_min - y_pad,
    y_max + y_pad,
)


# ============================================================
# Fit annotation
# ============================================================

ax.text(
    0.04,
    0.96,
    (
        rf"$p_c={pc:.5f}\pm{pc_err:.5f}$"
        "\n"
        rf"$\nu={nu:.3f}\pm{nu_err:.3f}$"
        "\n"
        rf"$\chi^2_\mathrm{{red}}={chi2_red:.3f}$"
    ),
    transform=ax.transAxes,
    ha="left",
    va="top",
)


# ============================================================
# Legend
# ============================================================

ax.legend(
    title=r"$d$",
    loc="lower right",
    ncol=2,
)


# ============================================================
# Save
# ============================================================

fig.savefig(
    FIG_PDF,
    bbox_inches="tight",
)

fig.savefig(
    FIG_PNG,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print()
print(f"Saved: {FIG_PDF}")
print(f"Saved: {FIG_PNG}")


if __name__ == "__main__":
    pass
