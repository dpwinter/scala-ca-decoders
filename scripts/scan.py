#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from pathlib import Path


def fmt_value(x: float | int | str) -> str:
    if isinstance(x, float):
        return f"{x:g}".replace(".", "p")
    return str(x)


def build_output_name(model: str, noise_model: str, params: dict) -> str:
    parts = [model, noise_model]

    ordered_keys = [
        "distance",
        "p_data",
        "p_meas",
        "p_signal",
        "U",
        "fN",
        "fC",
        "reset_time",
        "seed",
    ]

    for key in ordered_keys:
        if key in params:
            parts.append(f"{key}-{fmt_value(params[key])}")

    return "_".join(parts)


def run_cpp_job(exe: str, output_dir: Path, model: str, noise_model: str, params: dict, dry_run: bool) -> None:
    output_name = build_output_name(model, noise_model, params)
    output_path = output_dir / f"{output_name}.npy"

    cmd = [
        exe,
        "--model", model,
        "--noise-model", noise_model,
        "--distance", str(params["distance"]),
        "--samples", str(params["samples"]),
        "--max-steps", str(params["max_steps"]),
        "--p-data", str(params["p_data"]),
        "--seed", str(params["seed"]),
        "--output", str(output_path),
    ]

    if model == "scala1d" and noise_model == "phenomenological":
        cmd.extend([
            "--reset-time", str(params["reset_time"]),
            "--p-meas", str(params["p_meas"]),
            "--p-signal", str(params["p_signal"]),
        ])

    elif model == "scala1d" and noise_model == "code_capacity":
        pass

    elif model == "harrington1d" and noise_model == "phenomenological":
        cmd.extend([
            "--p-meas", str(params["p_meas"]),
            "--U", str(params["U"]),
            "--fN", str(params["fN"]),
            "--fC", str(params["fC"]),
        ])

    elif model == "harrington1d" and noise_model == "code_capacity":
        cmd.extend([
            "--U", str(params["U"]),
            "--fN", str(params["fN"]),
            "--fC", str(params["fC"]),
        ])

    else:
        raise ValueError(f"Unsupported C++ combination: model={model}, noise_model={noise_model}")

    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)

def is_valid_job(model: str, noise_model: str, params: dict) -> bool:
    # enforce reset_time always <= distance.
    if model in {"scala1d", "scala2d"} and noise_model == "phenomenological":
        if "reset_time" in params and params["reset_time"] > params["distance"]:
            return False
    return True

def run_python_job(module: str, output_dir: Path, model: str, noise_model: str, params: dict, dry_run: bool) -> None:
    cmd = [
        "python",
        "-m", module,
        "--noise-model", noise_model,
        "--distance", str(params["distance"]),
        "--p-data", str(params["p_data"]),
        "--samples", str(params["samples"]),
        "--seed", str(params["seed"]),
        "--output-dir", str(output_dir),
    ]

    if model in {"scala2d", "harrington2d"} and noise_model == "phenomenological":
        cmd.extend([
            "--p-meas", str(params["p_meas"]),
            "--p-signal", str(params["p_signal"]),
            "--reset-time", str(params["reset_time"]),
        ])

    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)

def expand_grid(grid: dict, linked: list[list[str]] | None = None) -> list[dict]:
    if not linked:
        keys = list(grid.keys())
        values = [grid[k] for k in keys]
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    # handle linked groups
    used = set()
    linked_jobs = []

    for group in linked:
        group_vals = [grid[k] for k in group]
        lengths = [len(v) for v in group_vals]
        if len(set(lengths)) != 1:
            raise ValueError(f"Linked parameters {group} must have same length")

        for i in range(lengths[0]):
            job = {}
            for k in group:
                job[k] = grid[k][i]
            linked_jobs.append(job)
        used.update(group)

    # remaining (unlinked) params
    remaining_keys = [k for k in grid if k not in used]

    if not remaining_keys:
        return linked_jobs

    remaining_vals = [grid[k] for k in remaining_keys]

    jobs = []
    for base in linked_jobs:
        for combo in itertools.product(*remaining_vals):
            job = base.copy()
            for k, v in zip(remaining_keys, combo):
                job[k] = v
            jobs.append(job)

    return jobs

def load_scan_config(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to scan JSON config")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_scan_config(Path(args.config))

    backend = cfg.get("backend", "cpp")
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model = cfg["model"]
    noise_model = cfg["noise_model"]
    # grid = cfg["grid"]
    jobs = expand_grid(cfg["grid"], cfg.get("linked"))

    print(f"Running {len(jobs)} jobs")

    skipped = 0

    if backend == "cpp":
        exe = cfg.get("executable", "./build/ca_run")
        for params in jobs:
            if not is_valid_job(model, noise_model, params):
                skipped += 1
                continue
            run_cpp_job(exe, output_dir, model, noise_model, params, args.dry_run)

        if skipped > 0:
            print(f"Skipped {skipped} invalid jobs (e.g. reset_time > distance)")
        return

    if backend == "python":
        module = cfg["module"]
        for params in jobs:
            if not is_valid_job(model, noise_model, params):
                skipped += 1
                continue
            run_python_job(module, output_dir, model, noise_model, params, args.dry_run)

        if skipped > 0:
            print(f"Skipped {skipped} invalid jobs (e.g. reset_time > distance)")
        return

    raise ValueError(f"Unsupported backend: {backend}")


if __name__ == "__main__":
    main()
