"""
Parallel-safe Harrington1D code-capacity runner.

Important:
    --shots is the TARGET TOTAL number of shots per point.

Example:

    python -m scripts.run_code_capacity \
        --d 9 27 81 \
        --p 0.02 0.03 0.04 0.05 0.10 0.20 \
        --shots 1000000 \
        --output data/code_capacity/code_capacity.csv

If a point already contains 100000 shots, only another
900000 shots are simulated.

Multiple processes may safely write to the same CSV.
"""

from pathlib import Path
from contextlib import contextmanager

import argparse
import csv
import fcntl
import hashlib
import math
import os
import tempfile

from src.harrington1d_optimized import simulate_code_capacity


FIELDNAMES = [
    "d",
    "p",
    "U",
    "fN",
    "fC",
    "max_steps",
    "shots",
    "failures",
    "logical_failures",
    "unresolved",
    "pL",
    "se",
]


# ============================================================
# Locks
# ============================================================

@contextmanager
def file_lock(path):
    """
    Exclusive inter-process lock associated with `path`.
    """

    lock_path = Path(
        str(path) + ".lock"
    )

    lock_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with lock_path.open("a") as lock:
        fcntl.flock(
            lock.fileno(),
            fcntl.LOCK_EX,
        )

        try:
            yield

        finally:
            fcntl.flock(
                lock.fileno(),
                fcntl.LOCK_UN,
            )


@contextmanager
def point_lock(
    output_file,
    key,
):
    """
    Lock one simulation point.

    Different (d,p) points can still run in parallel.
    """

    text = repr(key).encode()

    digest = hashlib.sha1(
        text
    ).hexdigest()[:16]

    lock_dir = (
        output_file.parent
        / ".run_locks"
    )

    lock_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    lock_file = (
        lock_dir
        / f"{output_file.stem}_{digest}"
    )

    with file_lock(lock_file):
        yield


# ============================================================
# Point identity
# ============================================================

def point_key(
    d,
    p,
    U,
    fN,
    fC,
    max_steps,
):
    return (
        int(d),
        round(float(p), 12),
        int(U),
        round(float(fN), 12),
        round(float(fC), 12),
        int(max_steps),
    )


# ============================================================
# CSV handling
# ============================================================

def read_rows(output_file):
    if not output_file.exists():
        return []

    if output_file.stat().st_size == 0:
        return []

    with output_file.open(
        "r",
        newline="",
    ) as file:

        return list(
            csv.DictReader(file)
        )


def row_key(row):
    return point_key(
        row["d"],
        row["p"],
        row["U"],
        row["fN"],
        row["fC"],
        row["max_steps"],
    )


def find_row(
    rows,
    key,
):
    for row in rows:
        if row_key(row) == key:
            return row

    return None


def empty_result(
    d,
    p,
    U,
    fN,
    fC,
    max_steps,
):
    return {
        "d": int(d),
        "p": float(p),

        "U": int(U),
        "fN": float(fN),
        "fC": float(fC),
        "max_steps": int(max_steps),

        "shots": 0,
        "failures": 0,
        "logical_failures": 0,
        "unresolved": 0,

        "pL": 0.0,
        "se": 0.0,
    }


def normalize_row(row):
    return {
        "d": int(row["d"]),
        "p": float(row["p"]),

        "U": int(row["U"]),
        "fN": float(row["fN"]),
        "fC": float(row["fC"]),
        "max_steps": int(
            row["max_steps"]
        ),

        "shots": int(row["shots"]),
        "failures": int(
            row["failures"]
        ),

        "logical_failures": int(
            row["logical_failures"]
        ),

        "unresolved": int(
            row["unresolved"]
        ),

        "pL": float(row["pL"]),
        "se": float(row["se"]),
    }


def recompute_statistics(row):
    shots = row["shots"]

    if shots == 0:
        row["pL"] = 0.0
        row["se"] = 0.0
        return row

    pL = (
        row["failures"]
        / shots
    )

    row["pL"] = pL

    row["se"] = math.sqrt(
        pL
        * (1.0 - pL)
        / shots
    )

    return row


def write_rows_atomic(
    output_file,
    rows,
):
    """
    Rewrite CSV atomically.

    Caller must hold the global file lock.
    """

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = sorted(
        rows,
        key=lambda row: (
            int(row["d"]),
            float(row["p"]),
        ),
    )

    fd, tmp_name = tempfile.mkstemp(
        prefix=output_file.name + ".",
        suffix=".tmp",
        dir=output_file.parent,
    )

    try:
        with os.fdopen(
            fd,
            "w",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=FIELDNAMES,
            )

            writer.writeheader()

            for row in rows:
                writer.writerow(row)

            file.flush()
            os.fsync(file.fileno())

        os.replace(
            tmp_name,
            output_file,
        )

    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass

        raise


def current_point(
    output_file,
    key,
    defaults,
):
    """
    Read current accumulated statistics.

    Must be called while holding the global CSV lock.
    """

    rows = read_rows(
        output_file
    )

    existing = find_row(
        rows,
        key,
    )

    if existing is None:
        return defaults

    return normalize_row(
        existing
    )


def merge_batch(
    output_file,
    key,
    defaults,
    batch,
):
    """
    Merge one newly simulated batch into the CSV.

    Must be called while holding the global CSV lock.
    """

    rows = read_rows(
        output_file
    )

    existing = find_row(
        rows,
        key,
    )

    if existing is None:
        combined = defaults.copy()

    else:
        combined = normalize_row(
            existing
        )

    combined["shots"] += int(
        batch["shots"]
    )

    combined["failures"] += int(
        batch["failures"]
    )

    combined[
        "logical_failures"
    ] += int(
        batch["logical_failures"]
    )

    combined["unresolved"] += int(
        batch["unresolved"]
    )

    recompute_statistics(
        combined
    )

    new_rows = [
        row
        for row in rows
        if row_key(row) != key
    ]

    new_rows.append(
        combined
    )

    write_rows_atomic(
        output_file,
        new_rows,
    )

    return combined


# ============================================================
# Deterministic batch seed
# ============================================================

def batch_seed(
    base_seed,
    d,
    p,
    starting_shots,
):
    """
    Different accumulated batches get different deterministic
    RNG streams.

    The seed depends on how many shots had already been
    accumulated before this batch.
    """

    p_int = int(
        round(
            float(p) * 10**12
        )
    )

    value = (
        int(base_seed)
        + 1_000_003 * int(d)
        + 1_000_000_007 * p_int
        + 10_000_019 * int(
            starting_shots
        )
    )

    return value % (
        2**63 - 1
    )


# ============================================================
# One Monte Carlo batch
# ============================================================

def run_batch(
    d,
    p,
    shots,
    U,
    fN,
    fC,
    max_steps,
    seed,
):
    return simulate_code_capacity(
        d=d,
        p=p,
        shots=shots,
        U=U,
        fN=fN,
        fC=fC,
        max_steps=max_steps,
        seed=seed,
        p_signal=0.0,
    )


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Parallel-safe Harrington1D "
            "code-capacity simulation."
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
        "--shots",
        type=int,
        default=100_000,
        help=(
            "Target TOTAL shots per point. "
            "Existing statistics are extended "
            "up to this value."
        ),
    )

    parser.add_argument(
        "--U",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--fN",
        type=float,
        default=0.4,
    )

    parser.add_argument(
        "--fC",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    output_file = Path(
        args.output
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.shots <= 0:
        raise ValueError(
            "--shots must be positive"
        )

    for d in args.d:

        print()
        print("=" * 72)
        print(f"d = {d}")
        print("=" * 72)

        for p in args.p:

            key = point_key(
                d,
                p,
                args.U,
                args.fN,
                args.fC,
                args.max_steps,
            )

            defaults = empty_result(
                d,
                p,
                args.U,
                args.fN,
                args.fC,
                args.max_steps,
            )

            # ------------------------------------------------
            # Lock this particular point for the entire batch.
            #
            # Other points can still run concurrently.
            # ------------------------------------------------

            with point_lock(
                output_file,
                key,
            ):

                # --------------------------------------------
                # Read current statistics safely.
                # --------------------------------------------

                with file_lock(
                    output_file
                ):
                    current = current_point(
                        output_file,
                        key,
                        defaults,
                    )

                existing_shots = int(
                    current["shots"]
                )

                target_shots = int(
                    args.shots
                )

                if (
                    existing_shots
                    >= target_shots
                ):
                    print(
                        f"d={d:3d}  "
                        f"p={p:.5f}  "
                        f"N={existing_shots:9d}  "
                        "TARGET REACHED -- skipping"
                    )

                    continue

                additional_shots = (
                    target_shots
                    - existing_shots
                )

                seed = batch_seed(
                    args.seed,
                    d,
                    p,
                    existing_shots,
                )

                print(
                    f"d={d:3d}  "
                    f"p={p:.5f}  "
                    f"existing={existing_shots:9d}  "
                    f"target={target_shots:9d}  "
                    f"adding={additional_shots:9d}",
                    flush=True,
                )

                # --------------------------------------------
                # Expensive Monte Carlo calculation.
                #
                # Global CSV lock is NOT held.
                # The per-point lock is held so another worker
                # cannot simulate this same point concurrently.
                # --------------------------------------------

                batch = run_batch(
                    d=d,
                    p=p,
                    shots=additional_shots,
                    U=args.U,
                    fN=args.fN,
                    fC=args.fC,
                    max_steps=args.max_steps,
                    seed=seed,
                )

                # --------------------------------------------
                # Merge batch atomically.
                # --------------------------------------------

                with file_lock(
                    output_file
                ):
                    combined = merge_batch(
                        output_file,
                        key,
                        defaults,
                        batch,
                    )

                print(
                    f"d={d:3d}  "
                    f"p={p:.5f}  "
                    f"N={combined['shots']:9d}  "
                    f"fail={combined['failures']:8d}  "
                    f"log={combined['logical_failures']:8d}  "
                    f"unres={combined['unresolved']:8d}  "
                    f"pL={combined['pL']:.6e}",
                    flush=True,
                )

    print()
    print(
        f"Saved results to {output_file}"
    )


if __name__ == "__main__":
    main()
