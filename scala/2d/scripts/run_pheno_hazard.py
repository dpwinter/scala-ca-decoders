"""
SCALA2D time-resolved first-passage hazard diagnostic.

This is the SCALA2D analogue of the SCALA1D hazard-scan runner used for the
stationarity figure.  It intentionally makes no burn-in/stationarity choice.
It records sufficient first-passage statistics in reset-period blocks, from
which h(t)=failures/exposure can be reconstructed and rebinned in raw CA time.

The actual SCALA2D dynamics, phenomenological-noise ordering, signal-noise
ordering, exact residual-syndrome MWPM checker, and exact staircase reset rule
are imported from scripts.run_pheno so this diagnostic cannot silently drift
away from the production runner.

Default noise model:
    q = p
    p_sig = 0

Reset schedule:
    t_R(d) = w0(d) + reset_offset,
where w0(d) is the exact discrete staircase used by scripts.run_pheno.

`--shots` is a TARGET TOTAL per (d,p,q,p_sig,t_R,rounds) configuration.
Existing samples in the CSV are reused; rerunning with a larger target adds an
independent continuation.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# IMPORTANT: import the exact production dynamics/checker implementation.
from scripts.run_pheno import (
    staircase_w0,
    probability_threshold,
    apply_round,
    build_matching,
    make_group,
    active_lanes,
    unpack_syndrome_into,
    actual_logicals_into,
)


BASE_SEED = 12345
DEFAULT_OUTPUT = Path("data/pheno/scala2d_pheno_hazard_scan.csv")


COLUMNS = [
    "d",
    "p",
    "p_meas",
    "p_sig",
    "w0",
    "reset_offset",
    "reset_period",
    "rounds",
    "block",
    "t_start",
    "t_end",
    "block_length",
    "shots",
    "at_risk",
    "failures",
    "survivors_end",
    "exposure",
    "sum_failure_t",
    "sum_failure_t2",
]

KEY_COLUMNS = [
    "d",
    "p",
    "p_meas",
    "p_sig",
    "w0",
    "reset_offset",
    "reset_period",
    "rounds",
    "block",
]

ADDITIVE_COLUMNS = [
    "shots",
    "at_risk",
    "failures",
    "survivors_end",
    "exposure",
    "sum_failure_t",
    "sum_failure_t2",
]


class FileLock:
    """Advisory lock on a separate .lock file (macOS/Linux)."""

    def __init__(self, path: Path):
        self.path = Path(str(path) + ".lock")
        self.handle = None

    def __enter__(self):
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        import fcntl

        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
        self.handle = None


def deterministic_seed(base_seed, d, p, q, p_sig, tR, rounds, existing_shots):
    text = (
        f"{int(base_seed)}|{int(d)}|{float(p):.12g}|{float(q):.12g}|"
        f"{float(p_sig):.12g}|{int(tR)}|{int(rounds)}|{int(existing_shots)}"
    )
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int(int.from_bytes(digest, "little") % (2**63 - 1))


def empty_dataframe():
    return pd.DataFrame(columns=COLUMNS)


def read_csv_safe(path: Path):
    if not path.exists():
        return empty_dataframe()
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return empty_dataframe()
    if df.empty:
        return empty_dataframe()
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise RuntimeError(
            f"{path} has incompatible schema; missing {sorted(missing)}"
        )
    return df[COLUMNS].copy()


def config_mask(df, d, p, q, p_sig, w0, reset_offset, tR, rounds):
    if df.empty:
        return np.zeros(0, dtype=bool)
    return (
        (df["d"].astype(int) == int(d))
        & np.isclose(df["p"].astype(float), float(p), atol=1e-12, rtol=0)
        & np.isclose(df["p_meas"].astype(float), float(q), atol=1e-12, rtol=0)
        & np.isclose(df["p_sig"].astype(float), float(p_sig), atol=1e-12, rtol=0)
        & (df["w0"].astype(int) == int(w0))
        & (df["reset_offset"].astype(int) == int(reset_offset))
        & (df["reset_period"].astype(int) == int(tR))
        & (df["rounds"].astype(int) == int(rounds))
    )


def existing_shots(output, d, p, q, p_sig, w0, reset_offset, tR, rounds):
    output = Path(output)
    with FileLock(output):
        df = read_csv_safe(output)
        sub = df.loc[
            config_mask(df, d, p, q, p_sig, w0, reset_offset, tR, rounds)
        ]
        if sub.empty:
            return 0
        block0 = sub[sub["block"].astype(int) == 0]
        if block0.empty:
            raise RuntimeError("Existing configuration is missing block 0")
        return int(block0.iloc[0]["shots"])


def aggregate_first_failures(
    first_failure,
    d,
    p,
    q,
    p_sig,
    w0,
    reset_offset,
    reset_period,
    rounds,
):
    first_failure = np.asarray(first_failure, dtype=np.int64)
    shots = len(first_failure)
    n_blocks = int(math.ceil(rounds / reset_period))
    rows = []

    for block in range(n_blocks):
        t_start = int(block * reset_period)
        t_end = int(min((block + 1) * reset_period, rounds))
        block_length = t_end - t_start

        alive_start = (first_failure < 0) | (first_failure > t_start)
        at_risk = int(np.count_nonzero(alive_start))
        if at_risk == 0:
            break

        failed_here = (first_failure > t_start) & (first_failure <= t_end)
        failures = int(np.count_nonzero(failed_here))
        survived_end = (first_failure < 0) | (first_failure > t_end)
        survivors_end = int(np.count_nonzero(survived_end))

        failure_times = first_failure[failed_here]
        exposure_failed = int(
            np.sum(failure_times - t_start, dtype=np.int64)
        )
        exposure_survivors = int(survivors_end * block_length)
        exposure = exposure_failed + exposure_survivors

        if failures:
            ft = failure_times.astype(np.int64, copy=False)
            sum_failure_t = int(np.sum(ft, dtype=np.int64))
            sum_failure_t2 = int(np.sum(ft * ft, dtype=np.int64))
        else:
            sum_failure_t = 0
            sum_failure_t2 = 0

        rows.append(
            {
                "d": int(d),
                "p": float(p),
                "p_meas": float(q),
                "p_sig": float(p_sig),
                "w0": int(w0),
                "reset_offset": int(reset_offset),
                "reset_period": int(reset_period),
                "rounds": int(rounds),
                "block": int(block),
                "t_start": int(t_start),
                "t_end": int(t_end),
                "block_length": int(block_length),
                "shots": int(shots),
                "at_risk": int(at_risk),
                "failures": int(failures),
                "survivors_end": int(survivors_end),
                "exposure": int(exposure),
                "sum_failure_t": int(sum_failure_t),
                "sum_failure_t2": int(sum_failure_t2),
            }
        )

    return pd.DataFrame(rows, columns=COLUMNS)


def simulate_first_failure(*, d, p, q, p_sig, reset_period, shots, rounds, seed):
    """Return first-failure round for every shot; -1 means censored at rounds."""

    threshold_data = probability_threshold(p)
    threshold_meas = probability_threshold(q)
    threshold_sig = probability_threshold(p_sig)

    matching = build_matching(d)

    groups = []
    group_offsets = []
    remaining = int(shots)
    offset = 0
    group_index = 0

    while remaining > 0:
        n_lanes = min(64, remaining)
        groups.append(
            make_group(
                d=d,
                n_lanes=n_lanes,
                seed=seed + 1000003 * group_index,
            )
        )
        group_offsets.append(offset)
        offset += n_lanes
        remaining -= n_lanes
        group_index += 1

    first_failure = np.full(shots, -1, dtype=np.int64)

    syndrome_buffer = np.empty((shots, d * d), dtype=np.uint8)
    actual_buffer = np.empty((shots, 2), dtype=np.uint8)
    map_group = np.empty(shots, dtype=np.int32)
    map_lane = np.empty(shots, dtype=np.int16)

    for round_number in range(1, rounds + 1):
        alive_count = sum(int(g["alive"]).bit_count() for g in groups)
        if alive_count == 0:
            break

        reset_now = (round_number % reset_period == 0)
        n_active = 0

        for gi, group in enumerate(groups):
            if group["alive"] == 0:
                continue

            apply_round(
                group["h"],
                group["v"],
                group["N"],
                group["W"],
                group["E"],
                group["S"],
                group["true_defect"],
                group["measured_defect"],
                group["new_N"],
                group["new_W"],
                group["new_E"],
                group["new_S"],
                d,
                group["alive"],
                group["rng_state"],
                threshold_data,
                threshold_meas,
                threshold_sig,
                reset_now,
            )

            lanes = active_lanes(group["alive"], group["n_lanes"])
            n = len(lanes)
            if n == 0:
                continue

            sl = slice(n_active, n_active + n)
            unpack_syndrome_into(
                group["true_defect"], lanes, syndrome_buffer[sl]
            )
            actual_logicals_into(
                group["h"], group["v"], lanes, actual_buffer[sl]
            )
            map_group[sl] = gi
            map_lane[sl] = lanes
            n_active += n

        if n_active == 0:
            break

        syndromes = syndrome_buffer[:n_active]
        actual = actual_buffer[:n_active]
        nonzero = np.any(syndromes != 0, axis=1)
        failed = np.zeros(n_active, dtype=bool)

        zero_rows = ~nonzero
        if np.any(zero_rows):
            failed[zero_rows] = np.any(actual[zero_rows] != 0, axis=1)

        nz = np.flatnonzero(nonzero)
        if len(nz) > 0:
            predicted = np.asarray(
                matching.decode_batch(syndromes[nz]), dtype=np.uint8
            )
            if predicted.ndim == 1:
                predicted = predicted[:, None]
            if predicted.shape[1] != 2:
                raise RuntimeError(
                    "PyMatching did not return two logical observables"
                )
            failed[nz] = np.any(predicted != actual[nz], axis=1)

        failed_rows = np.flatnonzero(failed)
        if len(failed_rows) == 0:
            continue

        failed_words = [np.uint64(0) for _ in groups]

        for row in failed_rows:
            gi = int(map_group[row])
            lane = int(map_lane[row])
            trajectory = group_offsets[gi] + lane
            first_failure[trajectory] = round_number
            failed_words[gi] |= np.uint64(1) << np.uint64(lane)

        for gi, word in enumerate(failed_words):
            if word != 0:
                groups[gi]["alive"] &= ~word

    return first_failure


def run_batch(task):
    (
        d,
        p,
        q,
        p_sig,
        w0,
        reset_offset,
        tR,
        rounds,
        batch_shots,
        seed,
    ) = task

    first_failure = simulate_first_failure(
        d=d,
        p=p,
        q=q,
        p_sig=p_sig,
        reset_period=tR,
        shots=batch_shots,
        rounds=rounds,
        seed=seed,
    )

    data = aggregate_first_failures(
        first_failure=first_failure,
        d=d,
        p=p,
        q=q,
        p_sig=p_sig,
        w0=w0,
        reset_offset=reset_offset,
        reset_period=tR,
        rounds=rounds,
    )

    n_failed = int(np.count_nonzero(first_failure > 0))
    n_censored = int(np.count_nonzero(first_failure < 0))
    mean_failed_time = (
        float(np.mean(first_failure[first_failure > 0]))
        if n_failed > 0
        else math.nan
    )

    return {
        "d": d,
        "p": p,
        "q": q,
        "p_sig": p_sig,
        "w0": w0,
        "reset_offset": reset_offset,
        "tR": tR,
        "rounds": rounds,
        "batch_shots": batch_shots,
        "n_failed": n_failed,
        "n_censored": n_censored,
        "mean_failed_time": mean_failed_time,
        "data": data,
    }


def merge_batch(output, batch_df):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with FileLock(output):
        current = read_csv_safe(output)

        if current.empty:
            merged = batch_df.copy()
        else:
            combined = pd.concat([current, batch_df], ignore_index=True)
            grouped_rows = []

            for _, sub in combined.groupby(KEY_COLUMNS, sort=False, dropna=False):
                row = sub.iloc[-1].copy()
                row["t_start"] = int(sub.iloc[0]["t_start"])
                row["t_end"] = int(sub.iloc[0]["t_end"])
                row["block_length"] = int(sub.iloc[0]["block_length"])
                for col in ADDITIVE_COLUMNS:
                    row[col] = int(sub[col].sum())
                grouped_rows.append(row)

            merged = pd.DataFrame(grouped_rows, columns=COLUMNS)

        merged = merged.sort_values(
            [
                "d",
                "p",
                "p_meas",
                "p_sig",
                "reset_period",
                "rounds",
                "block",
            ]
        ).reset_index(drop=True)

        fd, temp_name = tempfile.mkstemp(
            prefix=output.name + ".",
            suffix=".tmp",
            dir=str(output.parent),
        )
        os.close(fd)
        try:
            merged.to_csv(temp_name, index=False)
            os.replace(temp_name, output)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SCALA2D time-resolved first-passage hazard diagnostic."
    )

    parser.add_argument("--d", nargs="+", type=int, required=True)
    parser.add_argument("--p", nargs="+", type=float, required=True)
    parser.add_argument(
        "--p-meas",
        nargs="*",
        type=float,
        default=None,
        help="If omitted, q=p.",
    )
    parser.add_argument(
        "--p-sig",
        nargs="*",
        type=float,
        default=None,
        help="Signal-noise probabilities. Default 0.",
    )
    parser.add_argument(
        "--signal-equals-p",
        action="store_true",
        help="Use p_sig=p for every p point.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=200_000,
        help="TARGET TOTAL shots per configuration; existing data are reused.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Explicit common simulation length in raw CA rounds.",
    )
    parser.add_argument(
        "--rounds-factor",
        type=float,
        default=25.0,
        help="If --rounds is omitted: rounds=ceil(rounds_factor*d).",
    )
    parser.add_argument(
        "--reset-offset",
        type=int,
        default=0,
        help="Use t_R=w0(d)+offset. Production staircase uses 0.",
    )
    parser.add_argument(
        "--batch-shots",
        type=int,
        default=4096,
        help="Independent batch size submitted to each worker.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.shots <= 0 or args.batch_shots <= 0:
        raise ValueError("shots and batch-shots must be positive")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.rounds is None and args.rounds_factor <= 0:
        raise ValueError("rounds-factor must be positive")
    if args.signal_equals_p and args.p_sig:
        raise ValueError("Use either --signal-equals-p or --p-sig, not both")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    configs = []

    for d in args.d:
        w0 = staircase_w0(d)
        tR = w0 + args.reset_offset
        if tR < 1:
            raise ValueError(f"d={d}: t_R={tR} is invalid")

        rounds = int(args.rounds) if args.rounds is not None else int(
            math.ceil(args.rounds_factor * d)
        )

        for p in args.p:
            q_values = [p] if args.p_meas is None or len(args.p_meas) == 0 else args.p_meas

            if args.signal_equals_p:
                sig_values = [p]
            elif args.p_sig is None or len(args.p_sig) == 0:
                sig_values = [0.0]
            else:
                sig_values = args.p_sig

            for q in q_values:
                for p_sig in sig_values:
                    configs.append(
                        (int(d), float(p), float(q), float(p_sig), int(w0), int(tR), rounds)
                    )

    print()
    print("=" * 100)
    print("SCALA2D TIME-RESOLVED FIRST-PASSAGE HAZARD DIAGNOSTIC")
    print("=" * 100)
    print("reset schedule       = exact staircase t_R=w0(d)+offset")
    print(f"reset offset         = {args.reset_offset:+d}")
    print(f"target shots / point = {args.shots}")
    print(f"batch shots          = {args.batch_shots}")
    print(f"workers              = {args.workers}")
    print(f"output               = {output}")
    for d in args.d:
        w0 = staircase_w0(d)
        print(f"  d={d:3d}: w0={w0:3d}, t_R={w0 + args.reset_offset:3d}")
    print(f"configurations       = {len(configs)}")
    print("=" * 100)

    tasks = []

    for d, p, q, p_sig, w0, tR, rounds in configs:
        have = existing_shots(
            output, d, p, q, p_sig, w0, args.reset_offset, tR, rounds
        )
        missing = max(0, args.shots - have)

        print(
            f"d={d:3d} p={p:.6f} q={q:.6f} p_sig={p_sig:.6f} "
            f"tR={tR:3d} rounds={rounds:4d}: existing={have} missing={missing}"
        )

        batch_index = 0
        done = 0
        while done < missing:
            n = min(args.batch_shots, missing - done)
            seed = deterministic_seed(
                args.seed,
                d,
                p,
                q,
                p_sig,
                tR,
                rounds,
                have + done,
            )
            tasks.append(
                (
                    d,
                    p,
                    q,
                    p_sig,
                    w0,
                    args.reset_offset,
                    tR,
                    rounds,
                    n,
                    seed,
                )
            )
            done += n
            batch_index += 1

    if not tasks:
        print("Nothing to do: all requested configurations already reach target shots.")
        return

    completed = 0

    def report(result):
        nonlocal completed
        merge_batch(output, result["data"])
        completed += 1
        print(
            f"[{completed}/{len(tasks)}] d={result['d']:3d} "
            f"p={result['p']:.6f} q={result['q']:.6f} "
            f"p_sig={result['p_sig']:.6f} tR={result['tR']:3d} "
            f"batch={result['batch_shots']:6d} failed={result['n_failed']:6d} "
            f"censored={result['n_censored']:6d} "
            f"mean_failed_t={result['mean_failed_time']:.2f}",
            flush=True,
        )

    if args.workers == 1:
        for task in tasks:
            report(run_batch(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_batch, task) for task in tasks]
            for future in as_completed(futures):
                report(future.result())

    print()
    print("=" * 100)
    print(f"Saved: {output}")
    print("=" * 100)


if __name__ == "__main__":
    main()
