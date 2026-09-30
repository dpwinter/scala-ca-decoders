"""
SCALA2D code-capacity Monte Carlo runner.

Parallel-safe design
--------------------
Multiple processes may work on the same output CSV and even on the same
(d, p) point.

Each worker atomically reserves a finite chunk of shots in a persistent
claim ledger before simulation. Active claims count toward the requested
target, so two workers cannot reserve the same work.

Every reservation receives a unique claim_id. The Monte Carlo RNG seed is
derived from

    base_seed, d, p, claim_id

so concurrent workers never reuse the same random stream.

Expensive simulation happens WITHOUT holding the file lock.

Files
-----
Main results:
    <output>.csv

Lock:
    <output>.csv.lock

Persistent reservation ledger:
    <output>.csv.claims.csv

The claim ledger is intentionally retained after completion. This makes
claim IDs monotonic and prevents RNG-stream reuse even after crashes.

Decoder
-------
Uses src.scala2d_optimized with the paper reset schedule

    1, 2, ..., d, d-1, ..., 2, 1

for a total runtime d^2.

Classification
--------------
Both conventions are stored from the same Monte Carlo shots:

    pL_raw
        unresolved final syndrome counts as decoder failure

    pL
        unresolved final syndrome is completed by MWPM and then classified
        by final homology
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pymatching

from filelock import FileLock

from src.scala2d_optimized import (
    simulate_code_capacity_residuals,
    extract_lane,
    syndrome_array,
    logical_error,
)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run parallel-safe SCALA2D code-capacity Monte Carlo."
    )

    parser.add_argument(
        "--d",
        type=int,
        nargs="+",
        required=True,
        help="Code distances.",
    )

    parser.add_argument(
        "--p",
        type=float,
        nargs="+",
        required=True,
        help="Physical error probabilities.",
    )

    parser.add_argument(
        "--shots",
        type=int,
        required=True,
        help=(
            "Target TOTAL stored + currently claimed shots per (d,p). "
            "Existing samples are reused."
        ),
    )

    parser.add_argument(
        "--claim-chunk",
        type=int,
        default=1_000_000,
        help=(
            "Maximum shots reserved by one worker at once. "
            "Smaller values allow better parallel sharing of the same point. "
            "Default: 1000000."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Base RNG seed.",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Shared output CSV.",
    )

    parser.add_argument(
        "--no-verify-mwpm",
        action="store_true",
        help="Disable explicit post-MWPM syndrome verification.",
    )

    return parser.parse_args()


# =============================================================================
# PyMatching graph
# =============================================================================

_MATCHING_CACHE = {}


def _node(y, x, d):
    return int(y) * int(d) + int(x)


def build_toric_matching(d):
    """
    Build code-capacity MWPM graph for the d x d toric code.

    Syndrome convention:

        D[y,x]
        = h[y,x]
        ^ h[y,x+1]
        ^ v[y,x]
        ^ v[y+1,x]

    Therefore

        h[y,x] connects (y,x) <-> (y,x-1)
        v[y,x] connects (y,x) <-> (y-1,x)
    """
    d = int(d)
    d2 = d * d

    matching = pymatching.Matching()

    for y in range(d):
        for x in range(d):
            matching.add_edge(
                _node(y, x, d),
                _node(y, (x - 1) % d, d),
                fault_ids=y * d + x,
                weight=1.0,
            )

    for y in range(d):
        for x in range(d):
            matching.add_edge(
                _node(y, x, d),
                _node((y - 1) % d, x, d),
                fault_ids=d2 + y * d + x,
                weight=1.0,
            )

    return matching


def get_toric_matching(d):
    d = int(d)

    if d not in _MATCHING_CACHE:
        _MATCHING_CACHE[d] = build_toric_matching(d)

    return _MATCHING_CACHE[d]


# =============================================================================
# Statistics / RNG
# =============================================================================

def binomial_se(p, n):
    if n <= 0:
        return float("nan")

    return math.sqrt(
        max(
            0.0,
            p * (1.0 - p) / n,
        )
    )


def deterministic_claim_seed(
    base_seed,
    d,
    p,
    claim_id,
):
    """
    Unique deterministic RNG seed for a reserved claim.

    claim_id is globally unique within the persistent claim ledger.
    """
    payload = (
        f"{int(base_seed)}|"
        f"{int(d)}|"
        f"{float(p):.17g}|"
        f"{int(claim_id)}"
    ).encode("ascii")

    digest = hashlib.blake2b(
        payload,
        digest_size=8,
    ).digest()

    value = int.from_bytes(
        digest,
        byteorder="little",
        signed=False,
    )

    return int(
        value % 2_147_483_647
    )


def bit_count(x):
    return int(x).bit_count()


# =============================================================================
# MWPM completion
# =============================================================================

def mwpm_classify_residual(
    h,
    v,
    matching,
    verify=True,
):
    h = np.asarray(
        h,
        dtype=np.uint8,
    )

    v = np.asarray(
        v,
        dtype=np.uint8,
    )

    d = int(h.shape[0])

    syndrome = syndrome_array(
        h,
        v,
    )

    syndrome_weight = int(
        syndrome.sum()
    )

    if syndrome_weight == 0:
        return (
            logical_error(h, v),
            0,
            0,
        )

    if syndrome_weight % 2:
        raise RuntimeError(
            "Residual syndrome has odd defect parity."
        )

    correction = np.asarray(
        matching.decode(
            np.ascontiguousarray(
                syndrome.reshape(-1),
                dtype=np.uint8,
            )
        ),
        dtype=np.uint8,
    )

    d2 = d * d

    if correction.size != 2 * d2:
        raise RuntimeError(
            "Unexpected PyMatching correction size: "
            f"{correction.size}, expected {2*d2}"
        )

    h_corr = correction[:d2].reshape(
        d,
        d,
    )

    v_corr = correction[d2:].reshape(
        d,
        d,
    )

    h_final = h ^ h_corr
    v_final = v ^ v_corr

    if verify:
        final_syndrome = syndrome_array(
            h_final,
            v_final,
        )

        if np.any(final_syndrome):
            raise RuntimeError(
                "MWPM completion did not clear residual syndrome."
            )

    return (
        logical_error(
            h_final,
            v_final,
        ),
        int(correction.sum()),
        syndrome_weight,
    )


# =============================================================================
# One simulation chunk
# =============================================================================

def run_chunk(
    d,
    p,
    shots,
    seed,
    verify_mwpm=True,
):
    groups = simulate_code_capacity_residuals(
        d=d,
        p=p,
        shots=shots,
        seed=seed,
    )

    matching = get_toric_matching(d)

    resolved_shots = 0
    unresolved_syndromes = 0

    resolved_logical_failures = 0

    mwpm_checked = 0
    mwpm_corrected = 0
    mwpm_logical_failures = 0

    mwpm_correction_weight_sum = 0
    mwpm_syndrome_weight_sum = 0

    for group in groups:
        valid_mask = int(
            group["valid_mask"]
        )

        syndrome_mask = (
            int(group["syndrome_mask"])
            & valid_mask
        )

        logical_mask = (
            int(group["logical_mask"])
            & valid_mask
        )

        resolved_mask = (
            valid_mask
            & ~syndrome_mask
        )

        n_resolved = bit_count(
            resolved_mask
        )

        n_resolved_logical = bit_count(
            logical_mask
            & resolved_mask
        )

        resolved_shots += n_resolved
        resolved_logical_failures += (
            n_resolved_logical
        )

        unresolved_bits = syndrome_mask

        n_unresolved = bit_count(
            unresolved_bits
        )

        unresolved_syndromes += (
            n_unresolved
        )

        while unresolved_bits:
            lsb = (
                unresolved_bits
                & -unresolved_bits
            )

            lane = (
                lsb.bit_length() - 1
            )

            h = extract_lane(
                group["h"],
                lane,
            )

            v = extract_lane(
                group["v"],
                lane,
            )

            (
                is_logical,
                correction_weight,
                syndrome_weight,
            ) = mwpm_classify_residual(
                h,
                v,
                matching,
                verify=verify_mwpm,
            )

            mwpm_checked += 1

            mwpm_correction_weight_sum += (
                correction_weight
            )

            mwpm_syndrome_weight_sum += (
                syndrome_weight
            )

            if is_logical:
                mwpm_logical_failures += 1
            else:
                mwpm_corrected += 1

            unresolved_bits ^= lsb

    if (
        resolved_shots
        + unresolved_syndromes
        != shots
    ):
        raise RuntimeError(
            "resolved + unresolved != shots"
        )

    if (
        mwpm_checked
        != unresolved_syndromes
    ):
        raise RuntimeError(
            "mwpm_checked != unresolved_syndromes"
        )

    if (
        mwpm_corrected
        + mwpm_logical_failures
        != unresolved_syndromes
    ):
        raise RuntimeError(
            "MWPM accounting mismatch"
        )

    failures_raw = (
        resolved_logical_failures
        + unresolved_syndromes
    )

    failures_mwpm = (
        resolved_logical_failures
        + mwpm_logical_failures
    )

    return {
        "shots": int(shots),

        "resolved_shots": int(
            resolved_shots
        ),

        "unresolved_syndromes": int(
            unresolved_syndromes
        ),

        "resolved_logical_failures": int(
            resolved_logical_failures
        ),

        "mwpm_checked": int(
            mwpm_checked
        ),

        "mwpm_corrected": int(
            mwpm_corrected
        ),

        "mwpm_logical_failures": int(
            mwpm_logical_failures
        ),

        "failures_raw": int(
            failures_raw
        ),

        "failures_mwpm": int(
            failures_mwpm
        ),

        "mwpm_correction_weight_sum": int(
            mwpm_correction_weight_sum
        ),

        "mwpm_syndrome_weight_sum": int(
            mwpm_syndrome_weight_sum
        ),
    }


# =============================================================================
# Output CSV schema
# =============================================================================

COUNT_COLUMNS = [
    "shots",
    "resolved_shots",
    "unresolved_syndromes",
    "resolved_logical_failures",
    "mwpm_checked",
    "mwpm_corrected",
    "mwpm_logical_failures",
    "failures_raw",
    "failures_mwpm",
    "mwpm_correction_weight_sum",
    "mwpm_syndrome_weight_sum",
]


OUTPUT_COLUMNS = [
    "d",
    "p",
    "shots",

    "resolved_shots",
    "unresolved_syndromes",
    "resolved_logical_failures",

    "mwpm_checked",
    "mwpm_corrected",
    "mwpm_logical_failures",

    "failures_raw",
    "failures_mwpm",

    "pL_raw",
    "se_raw",

    "pL",
    "se",

    "p_unresolved",
    "p_mwpm_logical_given_unresolved",

    "mwpm_mean_correction_weight",
    "mwpm_mean_syndrome_weight",

    "decoder_steps",
]


def recompute_probabilities(row):
    shots = int(
        row["shots"]
    )

    failures_raw = int(
        row["failures_raw"]
    )

    failures_mwpm = int(
        row["failures_mwpm"]
    )

    unresolved = int(
        row["unresolved_syndromes"]
    )

    mwpm_checked = int(
        row["mwpm_checked"]
    )

    mwpm_logical = int(
        row["mwpm_logical_failures"]
    )

    pL_raw = (
        failures_raw / shots
        if shots > 0
        else float("nan")
    )

    pL = (
        failures_mwpm / shots
        if shots > 0
        else float("nan")
    )

    row["pL_raw"] = pL_raw
    row["se_raw"] = binomial_se(
        pL_raw,
        shots,
    )

    row["pL"] = pL
    row["se"] = binomial_se(
        pL,
        shots,
    )

    row["p_unresolved"] = (
        unresolved / shots
        if shots > 0
        else float("nan")
    )

    row[
        "p_mwpm_logical_given_unresolved"
    ] = (
        mwpm_logical / mwpm_checked
        if mwpm_checked > 0
        else 0.0
    )

    row[
        "mwpm_mean_correction_weight"
    ] = (
        row[
            "mwpm_correction_weight_sum"
        ] / mwpm_checked
        if mwpm_checked > 0
        else 0.0
    )

    row[
        "mwpm_mean_syndrome_weight"
    ] = (
        row[
            "mwpm_syndrome_weight_sum"
        ] / mwpm_checked
        if mwpm_checked > 0
        else 0.0
    )

    return row


def chunk_to_row(
    d,
    p,
    chunk,
):
    row = {
        "d": int(d),
        "p": float(p),
        **chunk,
        "decoder_steps": int(d) * int(d),
    }

    return recompute_probabilities(
        row
    )


def combine_rows(
    old,
    new,
):
    if (
        int(old["decoder_steps"])
        != int(new["decoder_steps"])
    ):
        raise RuntimeError(
            "Cannot merge different decoder runtimes."
        )

    combined = {
        "d": int(old["d"]),
        "p": float(old["p"]),
        "decoder_steps": int(
            old["decoder_steps"]
        ),
    }

    for col in COUNT_COLUMNS:
        old_value = (
            int(old[col])
            if (
                col in old
                and pd.notna(old[col])
            )
            else 0
        )

        new_value = (
            int(new[col])
            if (
                col in new
                and pd.notna(new[col])
            )
            else 0
        )

        combined[col] = (
            old_value
            + new_value
        )

    return recompute_probabilities(
        combined
    )


# =============================================================================
# Main CSV I/O
# =============================================================================

def load_output(path):
    if not path.exists():
        return pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

    df = pd.read_csv(
        path
    )

    optional_defaults = {
        "mwpm_correction_weight_sum": 0,
        "mwpm_syndrome_weight_sum": 0,
        "mwpm_mean_correction_weight": 0.0,
        "mwpm_mean_syndrome_weight": 0.0,
    }

    for col, default in optional_defaults.items():
        if col not in df.columns:
            df[col] = default

    missing = (
        set(OUTPUT_COLUMNS)
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Existing CSV uses incompatible schema.\n"
            f"Missing columns: {sorted(missing)}"
        )

    return df[
        OUTPUT_COLUMNS
    ].copy()


def find_row(
    df,
    d,
    p,
):
    if df.empty:
        return None

    mask = (
        (
            df["d"].astype(int)
            == int(d)
        )
        &
        (
            (
                df["p"].astype(float)
                - float(p)
            ).abs()
            < 1e-12
        )
    )

    indices = df.index[
        mask
    ].tolist()

    if len(indices) == 0:
        return None

    if len(indices) > 1:
        raise RuntimeError(
            f"Duplicate CSV rows for d={d}, p={p}"
        )

    return indices[0]


def atomic_write_csv(
    df,
    path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        str(path)
        + f".tmp.{os.getpid()}"
    )

    df.to_csv(
        tmp,
        index=False,
    )

    os.replace(
        tmp,
        path,
    )


def save_output(
    df,
    path,
):
    clean = (
        df[
            OUTPUT_COLUMNS
        ]
        .sort_values(
            ["d", "p"]
        )
        .reset_index(
            drop=True
        )
    )

    atomic_write_csv(
        clean,
        path,
    )


# =============================================================================
# Persistent claim ledger
# =============================================================================

CLAIM_COLUMNS = [
    "claim_id",
    "d",
    "p",
    "shots",
    "pid",
    "host",
    "status",
    "seed",
    "created_unix",
    "finished_unix",
]


def claims_path(output):
    return Path(
        str(output) + ".claims.csv"
    )


def load_claims(path):
    if not path.exists():
        return pd.DataFrame(
            columns=CLAIM_COLUMNS
        )

    df = pd.read_csv(
        path
    )

    missing = (
        set(CLAIM_COLUMNS)
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Claim ledger uses incompatible schema.\n"
            f"Missing columns: {sorted(missing)}"
        )

    return df[
        CLAIM_COLUMNS
    ].copy()


def save_claims(
    df,
    path,
):
    clean = (
        df[
            CLAIM_COLUMNS
        ]
        .sort_values("claim_id")
        .reset_index(drop=True)
    )

    atomic_write_csv(
        clean,
        path,
    )


def process_alive(
    pid,
    host,
):
    """
    Check whether a claimant process is still alive.

    Claims from another hostname are conservatively treated as alive.
    """
    current_host = socket.gethostname()

    if str(host) != current_host:
        return True

    try:
        pid = int(pid)
    except Exception:
        return False

    if pid <= 0:
        return False

    try:
        os.kill(
            pid,
            0,
        )
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

    return True


def clean_stale_claims(
    claims,
):
    """
    Mark dead local active claims as stale.

    Completed/stale claims are retained forever so claim_id values
    are never recycled.
    """
    if claims.empty:
        return claims, 0

    stale_count = 0

    for idx in claims.index:
        if str(
            claims.at[idx, "status"]
        ) != "active":
            continue

        if not process_alive(
            claims.at[idx, "pid"],
            claims.at[idx, "host"],
        ):
            claims.at[
                idx,
                "status",
            ] = "stale"

            claims.at[
                idx,
                "finished_unix",
            ] = time.time()

            stale_count += 1

    return claims, stale_count


def stored_shots(
    df,
    d,
    p,
):
    idx = find_row(
        df,
        d,
        p,
    )

    if idx is None:
        return 0

    return int(
        df.loc[
            idx,
            "shots",
        ]
    )


def active_claimed_shots(
    claims,
    d,
    p,
):
    if claims.empty:
        return 0

    mask = (
        (claims["status"] == "active")
        &
        (
            claims["d"].astype(int)
            == int(d)
        )
        &
        (
            (
                claims["p"].astype(float)
                - float(p)
            ).abs()
            < 1e-12
        )
    )

    if not np.any(mask):
        return 0

    return int(
        claims.loc[
            mask,
            "shots",
        ].astype(int).sum()
    )


def reserve_claim(
    output,
    lock,
    d,
    p,
    target,
    max_chunk,
    base_seed,
):
    """
    Atomically reserve up to max_chunk shots.

    Returns
    -------
    None
        if stored + active claims already reaches target

    dict
        reservation information otherwise
    """
    cpath = claims_path(
        output
    )

    with lock:
        df = load_output(
            output
        )

        claims = load_claims(
            cpath
        )

        claims, n_stale = clean_stale_claims(
            claims
        )

        stored = stored_shots(
            df,
            d,
            p,
        )

        claimed = active_claimed_shots(
            claims,
            d,
            p,
        )

        effective = (
            stored
            + claimed
        )

        missing = max(
            0,
            int(target)
            - effective,
        )

        if missing == 0:
            if n_stale:
                save_claims(
                    claims,
                    cpath,
                )

            return None, {
                "stored": stored,
                "claimed": claimed,
                "effective": effective,
                "stale_cleaned": n_stale,
            }

        n_claim = min(
            missing,
            int(max_chunk),
        )

        if claims.empty:
            claim_id = 1
        else:
            claim_id = (
                int(
                    claims[
                        "claim_id"
                    ].astype(int).max()
                )
                + 1
            )

        seed = deterministic_claim_seed(
            base_seed,
            d,
            p,
            claim_id,
        )

        claim = {
            "claim_id": int(
                claim_id
            ),
            "d": int(d),
            "p": float(p),
            "shots": int(
                n_claim
            ),
            "pid": int(
                os.getpid()
            ),
            "host": socket.gethostname(),
            "status": "active",
            "seed": int(seed),
            "created_unix": float(
                time.time()
            ),
            "finished_unix": np.nan,
        }

        claims = pd.concat(
            [
                claims,
                pd.DataFrame(
                    [claim]
                ),
            ],
            ignore_index=True,
        )

        save_claims(
            claims,
            cpath,
        )

        return claim, {
            "stored": stored,
            "claimed": claimed,
            "effective": effective,
            "stale_cleaned": n_stale,
        }


def commit_claim(
    output,
    lock,
    claim,
    new_row,
):
    """
    Atomically:

      1. merge completed counts into the latest output CSV
      2. mark this claim completed

    Therefore there is no state in which counts have been committed
    but the claim is still active.
    """
    cpath = claims_path(
        output
    )

    claim_id = int(
        claim["claim_id"]
    )

    with lock:
        df = load_output(
            output
        )

        claims = load_claims(
            cpath
        )

        match = (
            claims[
                "claim_id"
            ].astype(int)
            == claim_id
        )

        indices = claims.index[
            match
        ].tolist()

        if len(indices) != 1:
            raise RuntimeError(
                f"Cannot find unique claim_id={claim_id}"
            )

        claim_idx = indices[0]

        status = str(
            claims.at[
                claim_idx,
                "status",
            ]
        )

        if status != "active":
            raise RuntimeError(
                f"claim_id={claim_id} has status={status}, "
                "expected active"
            )

        idx = find_row(
            df,
            claim["d"],
            claim["p"],
        )

        if idx is None:
            combined = new_row

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [combined]
                    ),
                ],
                ignore_index=True,
            )

        else:
            old_row = (
                df.loc[
                    idx
                ].to_dict()
            )

            combined = combine_rows(
                old_row,
                new_row,
            )

            for col in OUTPUT_COLUMNS:
                df.at[
                    idx,
                    col,
                ] = combined[col]

        claims.at[
            claim_idx,
            "status",
        ] = "completed"

        claims.at[
            claim_idx,
            "finished_unix",
        ] = time.time()

        # Both writes happen while holding the same lock.
        save_output(
            df,
            output,
        )

        save_claims(
            claims,
            cpath,
        )

        return combined


def abandon_claim(
    output,
    lock,
    claim,
):
    """
    Mark a claim stale after a caught simulation exception.

    This permits the missing shots to be claimed again later with
    a new claim_id and therefore a fresh RNG stream.
    """
    cpath = claims_path(
        output
    )

    claim_id = int(
        claim["claim_id"]
    )

    with lock:
        claims = load_claims(
            cpath
        )

        match = (
            claims[
                "claim_id"
            ].astype(int)
            == claim_id
        )

        indices = claims.index[
            match
        ].tolist()

        if len(indices) != 1:
            return

        idx = indices[0]

        if str(
            claims.at[
                idx,
                "status",
            ]
        ) == "active":
            claims.at[
                idx,
                "status",
            ] = "stale"

            claims.at[
                idx,
                "finished_unix",
            ] = time.time()

            save_claims(
                claims,
                cpath,
            )


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    if args.shots <= 0:
        raise ValueError(
            "--shots must be positive"
        )

    if args.claim_chunk <= 0:
        raise ValueError(
            "--claim-chunk must be positive"
        )

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lock = FileLock(
        str(output) + ".lock"
    )

    verify_mwpm = (
        not args.no_verify_mwpm
    )

    for d in args.d:
        d = int(d)

        if d <= 1:
            raise ValueError(
                f"Invalid d={d}"
            )

        print()
        print("=" * 110)
        print(
            f"d = {d}"
        )
        print(
            "decoder          = SCALA2D paper schedule"
        )
        print(
            "schedule         = 1,2,...,d,...,2,1"
        )
        print(
            f"CA steps         = {d*d}"
        )
        print(
            "terminal readout = MWPM for unresolved syndromes"
        )
        print(
            f"target shots     = {args.shots}"
        )
        print(
            f"claim chunk      = {args.claim_chunk}"
        )
        print(
            f"output           = {output}"
        )
        print(
            f"claims           = {claims_path(output)}"
        )
        print("=" * 110)

        for p in args.p:
            p = float(p)

            if not 0.0 <= p <= 1.0:
                raise ValueError(
                    f"Invalid p={p}"
                )

            target = int(
                args.shots
            )

            while True:
                claim, progress = reserve_claim(
                    output=output,
                    lock=lock,
                    d=d,
                    p=p,
                    target=target,
                    max_chunk=args.claim_chunk,
                    base_seed=args.seed,
                )

                if (
                    progress[
                        "stale_cleaned"
                    ]
                    > 0
                ):
                    print(
                        f"d={d:3d} p={p:.5f}: "
                        f"cleaned "
                        f"{progress['stale_cleaned']} "
                        f"stale claim(s)"
                    )

                if claim is None:
                    print(
                        f"d={d:3d} "
                        f"p={p:.5f} "
                        f"N_stored={progress['stored']:9d} "
                        f"N_claimed={progress['claimed']:9d} "
                        f"N_target={target:9d} "
                        "complete/fully claimed"
                    )
                    break

                claim_id = int(
                    claim["claim_id"]
                )

                add = int(
                    claim["shots"]
                )

                seed = int(
                    claim["seed"]
                )

                print(
                    f"d={d:3d} "
                    f"p={p:.5f} "
                    f"claim={claim_id:6d} "
                    f"N_stored={progress['stored']:9d} "
                    f"N_active={progress['claimed']:9d} "
                    f"N_add={add:9d} "
                    f"seed={seed:10d} "
                    "running...",
                    flush=True,
                )

                try:
                    # -----------------------------------------------------
                    # Expensive simulation: NO FILE LOCK.
                    # -----------------------------------------------------
                    chunk = run_chunk(
                        d=d,
                        p=p,
                        shots=add,
                        seed=seed,
                        verify_mwpm=verify_mwpm,
                    )

                    new_row = chunk_to_row(
                        d,
                        p,
                        chunk,
                    )

                    # -----------------------------------------------------
                    # Locked atomic commit + claim completion.
                    # -----------------------------------------------------
                    combined = commit_claim(
                        output=output,
                        lock=lock,
                        claim=claim,
                        new_row=new_row,
                    )

                except BaseException:
                    abandon_claim(
                        output=output,
                        lock=lock,
                        claim=claim,
                    )
                    raise

                print(
                    f"d={d:3d} "
                    f"p={p:.5f} "
                    f"claim={claim_id:6d} "
                    f"N={int(combined['shots']):9d} "
                    f"syn={int(combined['unresolved_syndromes']):7d} "
                    f"mwpm_ok={int(combined['mwpm_corrected']):7d} "
                    f"mwpm_log={int(combined['mwpm_logical_failures']):7d} "
                    f"pL={combined['pL']:.8e} "
                    f"pL_raw={combined['pL_raw']:.8e}",
                    flush=True,
                )

    print()
    print("=" * 110)
    print(
        f"Finished. Shared results: {output}"
    )
    print(
        f"Claim ledger:            {claims_path(output)}"
    )
    print("=" * 110)


if __name__ == "__main__":
    main()
