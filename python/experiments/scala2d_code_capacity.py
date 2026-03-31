from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from python._native import scala2d
from python.experiments.harrington2d_common import (
    gen_err_conf,
    toric_code_z_logicals,
    toric_code_z_stabilisers,
)


def benchmark_code_capacity(
    d: int,
    p: float,
    n_samples: int,
    *,
    seed: int = 12345,
    output_dir: str | Path | None = None,
) -> tuple[int, np.ndarray]:
    """
    Close port of the legacy benchmarkCodeCapacity:

    - one-shot iid data noise
    - bounded decoding loop
    - increasing reset schedule
    - unresolved trajectories count as failures
    """
    rng = np.random.default_rng(seed)

    H = toric_code_z_stabilisers(d)
    logicals = toric_code_z_logicals(d)

    outcomes = np.zeros(n_samples, dtype=np.uint8)

    for n in range(n_samples):
        ca = scala2d.CA(d, seed + n)

        xframe = gen_err_conf(2 * d * d, p, rng)
        w = int(np.sum(xframe))

        log_err = True
        t_r = 2
        reset_interval = 1
        t_r_max = d * d + d - 2

        for i in range(2 * t_r_max):
            syndrome = (H @ xframe) % 2

            if np.sum(syndrome) == 0:
                logicals_flipped = (logicals @ xframe) % 2
                log_err = bool(np.sum(logicals_flipped) > 0)
                break

            ca.step(xframe, syndrome)

            if i == t_r:
                ca.reset()
                reset_interval += 1
                if reset_interval == d:
                    reset_interval = 3
                t_r = i + reset_interval

        # keep the same diagnostic spirit as the legacy code
        if log_err and w < (d + 1) // 2:
            print(f"UNCORRECT: w={w}")

        outcomes[n] = int(log_err)

    n_fail = int(np.sum(outcomes))

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = f"scala2d_code_capacity_d{d}_pdata{p:g}_seed{seed}".replace(".", "p")
        np.save(output_dir / f"{stem}.npy", outcomes)

        meta = {
            "model": "scala2d",
            "noise_model": "code_capacity",
            "observable": "logical_error_rate",
            "distance": d,
            "samples": n_samples,
            "p_data": p,
            "seed": seed,
            "logical_error_rate": float(np.mean(outcomes)),
            "data_file": str(output_dir / f"{stem}.npy"),
        }
        (output_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))

    return n_fail, outcomes
