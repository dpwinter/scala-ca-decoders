from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


REQUIRED_KEYS = {
        "model",
        "backend",
        "noise_model",
        "experiment",
        "distance",
        "samples",
        "seed",
        "noise",
        "output_dir"
}

NOISE_KEYS = {"data", "measurement", "signal"}

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    missing = REQUIRED_KEYS - set(cfg)
    if missing:
        raise ValueError(f"Missing required config keys: {sorted(missing)}")

    if not isinstance(cfg["noise"], dict):
        raise ValueError("Config key 'noise' must be a dictionary.")

    missing_noise = NOISE_KEYS - set(cfg["noise"])
    if missing_noise:
        raise ValueError(f"Missing required noise keys: {sorted(missing_noise)}")

    for key in NOISE_KEYS:
        value = cfg["noise"][key]
        if not isinstance(value, (int, float)):
            raise ValueError(f"Noise rate '{key}' must be numeric, got {type(value).__name__}.")
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Noise rate '{key}' must be in [0,1], got {value}.")

    return cfg

def build_command(cfg: dict) -> list[str]:
    """
    Minimal placeholder command builder

    For now this only returns a mock command so we can standardize
    the interface before touching any simulation code.
    """
    model = cfg["model"]
    backend = cfg["backend"]

    if backend == "cpp":
        if model == "scala1d":

            name = (
                f"{cfg['model']}_"
                f"d={cfg['distance']}_"
                f"pdata={cfg['noise']['data']}_"
                f"pmeas={cfg['noise']['measurement']}_"
                f"psig={cfg['noise']['signal']}_"
                f"seed={cfg['seed']}"
            )
            output_path = str(Path(cfg["output_dir"]) / f"{name}.npy")

            cmd = [
                "build/scala1d_run",
                "--distance", str(cfg["distance"]),
                "--samples", str(cfg["samples"]),
                "--max-steps", str(cfg.get("max_steps", 0)),
                "--reset-time", str(cfg.get("reset_time", 0)),
                "--p-data", str(cfg["noise"]["data"]),
                "--p-meas", str(cfg["noise"]["measurement"]),
                "--p-signal", str(cfg["noise"]["signal"]),
                "--seed", str(cfg["seed"]),
                "--output", output_path,
            ]
            return cmd

        raise ValueError(f"Unsupported cpp model: {model}")

    if backend == "python":
        raise NotImplementedError("Python backend support not added yet.")

    raise ValueError(f"Unsupported backend: {backend}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--dry-run", 
                        action="store_true", 
                        help="Print the command instead of executing it.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    cmd = build_command(cfg)

    print("Loaded config:")
    print(json.dumps(cfg, indent=2))
    print("\nCommand:")
    print(" ".join(cmd))

    exe_path = Path(cmd[0])
    if not exe_path.exists():
        raise FileNotFoundError(f"Executable not found: {exe_path}. Did you compile the C++ backend?")

    if not args.dry_run:
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
