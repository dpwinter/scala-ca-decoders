"""
Generic SCALA1D code-capacity runner.

Examples
--------

Broad scan:

    python -m scripts.run_code_capacity \
        --d 3 5 7 9 11 21 31 41 51 71 91 111 131 151 171 201 \
        --p 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.42 0.44 0.46 0.47 0.48 0.49 0.5 \
        --output data/code_capacity/code_capacity.csv

FSS scan:

    python -m scripts.run_code_capacity \
        --d 21 31 41 51 71 91 111 131 151 171 201 \
        --p 0.44 0.445 0.45 0.455 0.46 0.465 0.47 0.475 0.48 0.485 \
            0.49 0.495 0.5 0.505 0.51 0.515 0.52 0.525 0.53 0.535 \
            0.54 0.545 0.55 0.555 0.56 \
        --output data/code_capacity/code_capacity_fss.csv \
        --shots 100000

The output file is resumable:
existing (d, p) points are skipped and new points are appended.
"""

from pathlib import Path
import argparse
import csv
import math

import pandas as pd
from scipy.stats import binom

from src.scala1d_optimized import code_capacity_failures


BASE_SEED = 12345


# ============================================================
# Exact ML result
# ============================================================

def ml_probability(d, p):
    """Exact ML logical failure probability for odd repetition code."""

    w_min = (d + 1) // 2

    return float(
        binom.sf(
            w_min - 1,
            d,
            p,
        )
    )


def binomial_se(p_hat, shots):
    """Binomial standard error."""

    if shots == 0:
        return math.nan

    return math.sqrt(
        p_hat * (1.0 - p_hat) / shots
    )


# ============================================================
# Existing data
# ============================================================

def load_existing_points(output_file):

    if not output_file.exists():
        return set()

    data = pd.read_csv(output_file)

    existing = set()

    for _, row in data.iterrows():

        existing.add(
            (
                int(row["d"]),
                round(float(row["p"]), 12),
            )
        )

    return existing


# ============================================================
# Run one point
# ============================================================

def run_point(
    d,
    p,
    shots,
    point_id,
):
    seed = (
        BASE_SEED
        + 1_000_003 * point_id
    )

    failures, actual_shots = (
        code_capacity_failures(
            d=d,
            p=p,
            shots=shots,
            seed=seed,
        )
    )

    p_hat = failures / actual_shots
    se = binomial_se(
        p_hat,
        actual_shots,
    )

    p_exact = ml_probability(
        d,
        p,
    )

    print(
        f"d={d:3d}  "
        f"p={p:.5f}  "
        f"N={actual_shots:9d}  "
        f"fail={failures:8d}  "
        f"pL={p_hat:.6e}  "
        f"exact={p_exact:.6e}"
    )

    return {
        "d": d,
        "p": p,
        "shots": actual_shots,
        "failures": failures,
        "pL": p_hat,
        "se": se,
        "pL_exact": p_exact,
        "status": "done",
    }


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "SCALA1D code-capacity simulation."
        )
    )

    parser.add_argument(
        "--d",
        nargs="+",
        type=int,
        required=True,
        help="Code distances.",
    )

    parser.add_argument(
        "--p",
        nargs="+",
        type=float,
        required=True,
        help="Physical error probabilities.",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file.",
    )

    parser.add_argument(
        "--shots",
        type=int,
        default=100_000,
        help=(
            "Number of Monte Carlo shots per (d,p) point. "
            "Default: 100000."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=BASE_SEED,
        help="Base RNG seed.",
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    global BASE_SEED
    BASE_SEED = args.seed

    d_values = args.d
    p_values = args.p
    shots = args.shots

    output_file = Path(
        args.output
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "d",
        "p",
        "shots",
        "failures",
        "pL",
        "se",
        "pL_exact",
        "status",
    ]

    existing = load_existing_points(
        output_file
    )

    if existing:
        print(
            f"Found {len(existing)} existing "
            f"(d,p) points in {output_file}"
        )

    file_exists = output_file.exists()

    mode = (
        "a"
        if file_exists
        else "w"
    )

    with output_file.open(
        mode,
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        point_id = 0

        for d in d_values:

            if d % 2 == 0:
                raise ValueError(
                    f"d must be odd, got {d}"
                )

            print()
            print("=" * 72)
            print(f"d = {d}")
            print("=" * 72)

            for p in p_values:

                if not 0.0 <= p <= 1.0:
                    raise ValueError(
                        f"p must lie in [0,1], got {p}"
                    )

                key = (
                    int(d),
                    round(float(p), 12),
                )

                if key in existing:

                    print(
                        f"d={d:3d}  "
                        f"p={p:.5f}  "
                        f"EXISTS -- skipping"
                    )

                    point_id += 1
                    continue

                result = run_point(
                    d=d,
                    p=p,
                    shots=shots,
                    point_id=point_id,
                )

                writer.writerow(
                    result
                )

                file.flush()

                existing.add(key)

                point_id += 1

    print()
    print(
        f"Saved results to {output_file}"
    )


if __name__ == "__main__":
    main()
