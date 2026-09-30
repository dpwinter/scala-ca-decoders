"""Safe Harrington2D phenomenological first-passage lifetime runner.

Noise order per CA round:
  1) data noise p
  2) exact syndrome
  3) measurement noise q on decoder syndrome
  4) one complete Harrington CA update
  5) independent CountSignal bit flips with rate p_count
  6) independent FlipSignal bit flips with rate p_flip
  7) exact post-update syndrome
  8) ideal PyMatching logical-sector test

Models:
  baseline: p=q=r, p_count=p_flip=0
  count:    p=q=p_count=r, p_flip=0
  full:     p=q=p_count=p_flip=r

CSV behavior is safe by default: an existing file is loaded, existing points are
skipped, and completed points are atomically upserted. Use --rerun-existing to
replace requested existing points, or --overwrite to deliberately discard the
whole output file.
"""

from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pymatching
from numba import njit

from src.harrington2d_optimized import (
    MASK64,
    N_DIR,
    N_CARD,
    _bernoulli_word,
    _build_geometry,
    _seed_rng,
    _step,
    _syndrome,
)

UINT32_SCALE = 4294967296.0


def probability_threshold(p: float) -> int:
    if p <= 0.0:
        return 0
    if p >= 1.0:
        return 0x100000000
    return int(math.floor(p * UINT32_SCALE))


# -----------------------------------------------------------------------------
# Signal noise: defined locally so the runner cannot silently pick up a stale
# or incompatible _apply_signal_noise() from src.harrington2d_optimized.
# -----------------------------------------------------------------------------

@njit
def apply_signal_noise(
    count_sig,
    flip_sig,
    count_threshold,
    flip_threshold,
    active_mask,
    rng_state,
    d,
    n_levels,
):
    if count_threshold != 0:
        for level in range(n_levels):
            for y in range(d):
                for x in range(d):
                    for direction in range(N_DIR):
                        count_sig[level, y, x, direction] ^= (
                            _bernoulli_word(rng_state, count_threshold)
                            & active_mask
                        )

    if flip_threshold != 0:
        for level in range(n_levels):
            for y in range(d):
                for x in range(d):
                    for direction in range(N_CARD):
                        flip_sig[level, y, x, direction] ^= (
                            _bernoulli_word(rng_state, flip_threshold)
                            & active_mask
                        )


def signal_noise_self_test() -> None:
    """Fail loudly if CountSignal/FlipSignal noise is not actually wired in."""
    d = 3
    n_levels = 1
    active = np.uint64(0b1111)
    cs = np.zeros((n_levels, d, d, N_DIR), dtype=np.uint64)
    fs = np.zeros((n_levels, d, d, N_CARD), dtype=np.uint64)

    rng = _seed_rng(0x5A17)
    apply_signal_noise(cs, fs, 0x100000000, 0, active, rng, d, n_levels)
    if not np.all((cs & active) == active) or np.any(fs != 0):
        raise RuntimeError("CountSignal-noise self-test failed")

    cs.fill(0)
    fs.fill(0)
    rng = _seed_rng(0x5A18)
    apply_signal_noise(cs, fs, 0, 0x100000000, active, rng, d, n_levels)
    if np.any(cs != 0) or not np.all((fs & active) == active):
        raise RuntimeError("FlipSignal-noise self-test failed")


# -----------------------------------------------------------------------------
# Toric-code MWPM checker
# -----------------------------------------------------------------------------

def node(y: int, x: int, d: int) -> int:
    return y * d + x


def build_matching(d: int) -> pymatching.Matching:
    matching = pymatching.Matching()

    # horizontal[y,x] touches checks (y,x) and (y,x-1)
    for y in range(d):
        for x in range(d):
            xm = x - 1 if x > 0 else d - 1
            matching.add_edge(
                node(y, x, d),
                node(y, xm, d),
                fault_ids={0} if x == 0 else set(),
                weight=1.0,
            )

    # vertical[y,x] touches checks (y,x) and (y+1,x)
    for y in range(d):
        yp = y + 1 if y + 1 < d else 0
        for x in range(d):
            matching.add_edge(
                node(y, x, d),
                node(yp, x, d),
                fault_ids={1} if y == 0 else set(),
                weight=1.0,
            )

    return matching


@njit
def apply_noise_and_measure(
    horizontal,
    vertical,
    true_defects,
    measured_defects,
    p_threshold,
    q_threshold,
    alive,
    rng_state,
    d,
):
    if p_threshold != 0:
        for y in range(d):
            for x in range(d):
                horizontal[y, x] ^= (
                    _bernoulli_word(rng_state, p_threshold) & alive
                )
                vertical[y, x] ^= (
                    _bernoulli_word(rng_state, p_threshold) & alive
                )

    _syndrome(horizontal, vertical, true_defects, d)

    for y in range(d):
        for x in range(d):
            measured_defects[y, x] = true_defects[y, x]
            if q_threshold != 0:
                measured_defects[y, x] ^= (
                    _bernoulli_word(rng_state, q_threshold) & alive
                )


def make_group(d, n_lanes, geometry, seed):
    n_levels = geometry["n_levels"]
    rep_count = geometry["rep_count"]
    max_bits = geometry["max_bits"]
    valid_mask = MASK64 if n_lanes == 64 else np.uint64((1 << n_lanes) - 1)

    count_sig = np.zeros((n_levels, d, d, N_DIR), dtype=np.uint64)
    flip_sig = np.zeros((n_levels, d, d, N_CARD), dtype=np.uint64)

    return {
        "horizontal": np.zeros((d, d), dtype=np.uint64),
        "vertical": np.zeros((d, d), dtype=np.uint64),
        "true_defects": np.zeros((d, d), dtype=np.uint64),
        "measured_defects": np.zeros((d, d), dtype=np.uint64),
        "count_sig": count_sig,
        "flip_sig": flip_sig,
        "new_count_sig": np.zeros_like(count_sig),
        "new_flip_sig": np.zeros_like(flip_sig),
        "counters": np.zeros((9, max_bits, rep_count), dtype=np.uint64),
        "ages": np.zeros(n_levels, dtype=np.int64),
        "rng_state": _seed_rng(int(seed)),
        "alive": valid_mask,
        "n_lanes": int(n_lanes),
    }


def active_lanes(alive, n_lanes):
    value = int(alive)
    return np.asarray(
        [lane for lane in range(n_lanes) if (value >> lane) & 1],
        dtype=np.int64,
    )


def unpack_syndrome_into(defects, lanes, target):
    flat = defects.reshape(-1)
    shifts = lanes.astype(np.uint64)[:, None]
    target[:, :] = (((flat[None, :] >> shifts) & np.uint64(1))).astype(np.uint8)


def actual_logicals_into(horizontal, vertical, lanes, target):
    logical_h = np.uint64(0)
    logical_v = np.uint64(0)
    for y in range(horizontal.shape[0]):
        logical_h ^= horizontal[y, 0]
    for x in range(vertical.shape[1]):
        logical_v ^= vertical[0, x]

    shifts = lanes.astype(np.uint64)
    target[:, 0] = ((logical_h >> shifts) & np.uint64(1)).astype(np.uint8)
    target[:, 1] = ((logical_v >> shifts) & np.uint64(1)).astype(np.uint8)


def advance_group(
    state,
    geometry,
    d,
    p_threshold,
    q_threshold,
    count_threshold,
    flip_threshold,
):
    alive = state["alive"]
    if alive == 0:
        return

    apply_noise_and_measure(
        state["horizontal"],
        state["vertical"],
        state["true_defects"],
        state["measured_defects"],
        p_threshold,
        q_threshold,
        alive,
        state["rng_state"],
        d,
    )

    _step(
        state["horizontal"],
        state["vertical"],
        state["measured_defects"],
        state["count_sig"],
        state["flip_sig"],
        state["new_count_sig"],
        state["new_flip_sig"],
        state["counters"],
        geometry["site_to_rep"],
        geometry["roles"],
        state["ages"],
        geometry["U_levels"],
        geometry["Q_levels"],
        geometry["periods"],
        geometry["nbits_levels"],
        geometry["threshold_N"],
        geometry["threshold_C"],
        alive,
        state["rng_state"],
        d,
        geometry["n_levels"],
        True,
    )

    # IMPORTANT: corrupt the persistent signal memories after the complete CA
    # update, exactly once per CA round.
    apply_signal_noise(
        state["count_sig"],
        state["flip_sig"],
        count_threshold,
        flip_threshold,
        alive,
        state["rng_state"],
        d,
        geometry["n_levels"],
    )

    _syndrome(state["horizontal"], state["vertical"], state["true_defects"], d)


def rates_for_model(model, r):
    if model == "baseline":
        return r, r, 0.0, 0.0
    if model == "count":
        return r, r, r, 0.0
    if model == "full":
        return r, r, r, r
    raise ValueError(f"unknown model {model!r}")


def simulate_point(
    d,
    p,
    q,
    p_count,
    p_flip,
    shots,
    max_rounds,
    walker_chunk,
    U,
    fN,
    fC,
    seed,
):
    geometry = _build_geometry(d, U, fN, fC)
    matching = build_matching(d)

    p_threshold = probability_threshold(p)
    q_threshold = probability_threshold(q)
    count_threshold = probability_threshold(p_count)
    flip_threshold = probability_threshold(p_flip)

    all_failure_times = []
    total_censored = 0
    completed = 0

    while completed < shots:
        chunk_shots = min(walker_chunk, shots - completed)
        groups = []
        group_offsets = []
        offset = 0
        group_number = 0

        while offset < chunk_shots:
            n_lanes = min(64, chunk_shots - offset)
            group_seed = seed + 1000003 * (completed // 64 + group_number)
            groups.append(make_group(d, n_lanes, geometry, group_seed))
            group_offsets.append(offset)
            offset += n_lanes
            group_number += 1

        chunk_failure_times = np.zeros(chunk_shots, dtype=np.int64)
        syndrome_buffer = np.empty((chunk_shots, d * d), dtype=np.uint8)
        actual_buffer = np.empty((chunk_shots, 2), dtype=np.uint8)
        map_group = np.empty(chunk_shots, dtype=np.int32)
        map_lane = np.empty(chunk_shots, dtype=np.int16)

        for t in range(1, max_rounds + 1):
            n_active = 0

            for group_index, state in enumerate(groups):
                if state["alive"] == 0:
                    continue

                advance_group(
                    state,
                    geometry,
                    d,
                    p_threshold,
                    q_threshold,
                    count_threshold,
                    flip_threshold,
                )

                lanes = active_lanes(state["alive"], state["n_lanes"])
                n = len(lanes)
                if n == 0:
                    continue

                sl = slice(n_active, n_active + n)
                unpack_syndrome_into(state["true_defects"], lanes, syndrome_buffer[sl])
                actual_logicals_into(
                    state["horizontal"], state["vertical"], lanes, actual_buffer[sl]
                )
                map_group[sl] = group_index
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
                    raise RuntimeError("PyMatching did not return two observables")
                failed[nz] = np.any(predicted != actual[nz], axis=1)

            failed_rows = np.flatnonzero(failed)
            if len(failed_rows) > 0:
                failed_words = [np.uint64(0) for _ in groups]
                for row in failed_rows:
                    gi = int(map_group[row])
                    lane = int(map_lane[row])
                    failed_words[gi] |= np.uint64(1) << np.uint64(lane)
                    chunk_failure_times[group_offsets[gi] + lane] = t

                for gi, word in enumerate(failed_words):
                    if word != 0:
                        groups[gi]["alive"] &= ~word

        censored_mask = chunk_failure_times == 0
        total_censored += int(np.count_nonzero(censored_mask))
        if np.any(~censored_mask):
            all_failure_times.append(chunk_failure_times[~censored_mask])
        completed += chunk_shots

    if all_failure_times:
        failure_times = np.concatenate(all_failure_times).astype(np.float64)
    else:
        failure_times = np.empty(0, dtype=np.float64)

    failures = int(len(failure_times))
    resolved = total_censored == 0

    if failures > 0:
        mean_failed_tf = float(np.mean(failure_times))
        min_tf = int(np.min(failure_times))
        max_tf_observed = int(np.max(failure_times))
    else:
        mean_failed_tf = math.nan
        min_tf = math.nan
        max_tf_observed = math.nan

    if resolved and failures == shots:
        mean_tf = float(np.mean(failure_times))
        std_tf = float(np.std(failure_times, ddof=0))
        se_tf = float(std_tf / math.sqrt(shots))
    else:
        mean_tf = math.nan
        std_tf = math.nan
        se_tf = math.nan

    return {
        "shots": int(shots),
        "failures": failures,
        "censored": int(total_censored),
        "resolved": bool(resolved),
        "mean_TF": mean_tf,
        "std_TF": std_tf,
        "se_TF": se_tf,
        "mean_failed_TF": mean_failed_tf,
        "min_TF": min_tf,
        "max_TF_observed": max_tf_observed,
    }


def run_point_worker(job):
    model = job["model"]
    d = job["d"]
    r = job["r"]
    p, q, p_count, p_flip = rates_for_model(model, r)

    stats = simulate_point(
        d=d,
        p=p,
        q=q,
        p_count=p_count,
        p_flip=p_flip,
        shots=job["shots"],
        max_rounds=job["max_rounds"],
        walker_chunk=job["walker_chunk"],
        U=job["U"],
        fN=job["fN"],
        fC=job["fC"],
        seed=job["seed"],
    )

    return {
        "model": model,
        "d": int(d),
        "r": float(r),
        "p": float(p),
        "q": float(q),
        "p_count": float(p_count),
        "p_flip": float(p_flip),
        "U": int(job["U"]),
        "fN": float(job["fN"]),
        "fC": float(job["fC"]),
        "max_rounds": int(job["max_rounds"]),
        **stats,
    }


OUTPUT_COLUMNS = [
    "model", "d", "r", "p", "q", "p_count", "p_flip",
    "U", "fN", "fC", "max_rounds", "shots", "failures", "censored",
    "resolved", "mean_TF", "std_TF", "se_TF", "mean_failed_TF",
    "min_TF", "max_TF_observed",
]
KEY_COLUMNS = ["model", "d", "r", "U", "fN", "fC"]


def atomic_write_csv(df, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(output)


def normalize_output(df):
    if len(df) == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    out = df.copy()
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = out[OUTPUT_COLUMNS]
    out = out.drop_duplicates(subset=KEY_COLUMNS, keep="last")
    return out.sort_values(["model", "d", "r"]).reset_index(drop=True)


def upsert_rows(existing, new_rows):
    current = pd.DataFrame(new_rows)
    if existing is None or len(existing) == 0:
        return normalize_output(current)
    if len(current) == 0:
        return normalize_output(existing)
    return normalize_output(pd.concat([existing, current], ignore_index=True))


def point_key(model, d, r, U, fN, fC):
    return (str(model), int(d), float(r), int(U), float(fN), float(fC))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", nargs="+", type=int, required=True)
    parser.add_argument("--r", nargs="+", type=float, required=True)
    parser.add_argument(
        "--models", nargs="+", choices=["baseline", "count", "full"],
        default=["baseline", "count", "full"],
    )
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--max-rounds", type=int, default=10_000_000)
    parser.add_argument("--walker-chunk", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--U", type=int, default=16)
    parser.add_argument("--fN", type=float, default=0.4)
    parser.add_argument("--fC", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/pheno/harrington2d_pheno_matched.csv"),
    )
    parser.add_argument(
        "--rerun-existing", action="store_true",
        help="Recompute requested points and replace only those rows.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="DANGEROUS: explicitly discard the entire existing output CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    signal_noise_self_test()

    for d in args.d:
        if d <= 0 or d % 2 == 0:
            raise ValueError(f"d must be positive and odd; got {d}")
    for r in args.r:
        if not 0.0 <= r <= 1.0:
            raise ValueError(f"r must lie in [0,1]; got {r}")
    if args.workers < 1:
        raise ValueError("--workers must be >=1")
    if args.walker_chunk < 64:
        raise ValueError("--walker-chunk must be >=64")

    if args.output.exists() and not args.overwrite:
        existing = normalize_output(pd.read_csv(args.output))
    else:
        existing = pd.DataFrame(columns=OUTPUT_COLUMNS)
        if args.overwrite and args.output.exists():
            print(f"WARNING: discarding existing file {args.output}")

    existing_keys = {
        point_key(row.model, row.d, row.r, row.U, row.fN, row.fC)
        for row in existing.itertuples(index=False)
    }

    jobs = []
    point_index = 0
    skipped = 0

    for model in args.models:
        for d in args.d:
            for r in args.r:
                key = point_key(model, d, r, args.U, args.fN, args.fC)
                if key in existing_keys and not args.rerun_existing:
                    skipped += 1
                    continue

                point_index += 1
                jobs.append({
                    "model": model,
                    "d": d,
                    "r": r,
                    "shots": args.shots,
                    "max_rounds": args.max_rounds,
                    "walker_chunk": args.walker_chunk,
                    "U": args.U,
                    "fN": args.fN,
                    "fC": args.fC,
                    "seed": (
                        args.seed
                        + 10000019 * point_index
                        + 1009 * d
                        + int(round(1e9 * r))
                    ),
                })

    print("=" * 100)
    print("SAFE HARRINGTON2D PHENOMENOLOGICAL FIRST-PASSAGE LIFETIME SCAN")
    print("=" * 100)
    print(f"models          = {args.models}")
    print(f"distances       = {args.d}")
    print(f"rates           = {args.r}")
    print(f"shots/point     = {args.shots}")
    print(f"workers         = {args.workers}")
    print(f"output          = {args.output}")
    print(f"existing rows   = {len(existing)}")
    print(f"skipped points  = {skipped}")
    print(f"points to run   = {len(jobs)}")
    print("signal-noise self-test: PASS")
    print("=" * 100)

    completed_rows = []

    def checkpoint():
        out = upsert_rows(existing, completed_rows)
        atomic_write_csv(out, args.output)

    if args.workers == 1:
        for i, job in enumerate(jobs, start=1):
            row = run_point_worker(job)
            completed_rows.append(row)
            if row["resolved"]:
                print(
                    f"[{i}/{len(jobs)}] {row['model']} d={row['d']} "
                    f"r={row['r']:.6g}: TF={row['mean_TF']:.6g} +- {row['se_TF']:.3g}"
                )
            else:
                print(
                    f"[{i}/{len(jobs)}] {row['model']} d={row['d']} "
                    f"r={row['r']:.6g}: UNRESOLVED ({row['censored']} censored)"
                )
            checkpoint()
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_job = {
                executor.submit(run_point_worker, job): job for job in jobs
            }
            done = 0
            for future in as_completed(future_to_job):
                row = future.result()
                completed_rows.append(row)
                done += 1
                if row["resolved"]:
                    print(
                        f"[{done}/{len(jobs)}] {row['model']} d={row['d']} "
                        f"r={row['r']:.6g}: TF={row['mean_TF']:.6g} +- {row['se_TF']:.3g}"
                    )
                else:
                    print(
                        f"[{done}/{len(jobs)}] {row['model']} d={row['d']} "
                        f"r={row['r']:.6g}: UNRESOLVED ({row['censored']} censored)"
                    )
                checkpoint()

    checkpoint()
    print(f"Saved safely: {args.output}")


if __name__ == "__main__":
    main()
