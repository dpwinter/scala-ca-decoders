from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from python._native import harrington2d as harrington
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
    Close port of the legacy benchmark_code_capacity:

    - one-shot iid data noise on the 2 d^2 qubits
    - run Harrington2D until the syndrome vanishes
    - evaluate logical failure from the final frame
    """
    rng = np.random.default_rng(seed)

    H = toric_code_z_stabilisers(d)
    logicals = toric_code_z_logicals(d)

    outcomes = np.zeros(n_samples, dtype=np.uint8)

    for i in range(n_samples):
        ca = harrington.CA(d, seed + i)

        xframe = gen_err_conf(2 * d * d, p, rng)
        syndrome = (H @ xframe) % 2

        while np.sum(syndrome) != 0:
            syndrome = (H @ xframe) % 2
            ca.step(xframe, syndrome)

        logicals_flipped = (logicals @ xframe) % 2
        outcomes[i] = int(np.sum(logicals_flipped) > 0)

    n_fail = int(np.sum(outcomes))

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = f"harrington2d_code_capacity_d{d}_pdata{p:g}_seed{seed}".replace(".", "p")
        np.save(output_dir / f"{stem}.npy", outcomes)

        meta = {
            "model": "harrington2d",
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
