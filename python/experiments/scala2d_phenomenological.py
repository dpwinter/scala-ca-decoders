from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pymatching import Matching

from python._native import scala2d
from python.experiments.harrington2d_common import (
    gen_err_conf,
    toric_code_z_logicals,
    toric_code_z_stabilisers,
)


def benchmark_pheno(
    d: int,
    p: float,
    n_samples: int,
    *,
    seed: int = 12345,
    output_dir: str | Path | None = None,
    p_meas: float = 0.0,
    p_sig: float = 0.0,
    reset_time: int | None = None,
) -> tuple[int, np.ndarray]:
    rng = np.random.default_rng(seed)

    H = toric_code_z_stabilisers(d)
    logicals = toric_code_z_logicals(d)
    matching = Matching.from_check_matrix(H, weights=np.log((1 - p) / p), faults_matrix=logicals)

    lifetimes = np.zeros(n_samples, dtype=np.int64)

    for i in range(n_samples):
        ca = scala2d.CA(d, seed + i)
        xframe = np.zeros(2 * d * d, dtype=bool)

        t = 0
        while True:
            t += 1

            xframe = xframe ^ gen_err_conf(2 * d * d, p, rng)
            syndrome = (H @ xframe) % 2

            predicted_logicals_flipped = matching.decode(syndrome)
            actual_logicals_flipped = (logicals @ xframe) % 2
            if not np.array_equal(predicted_logicals_flipped, actual_logicals_flipped):
                break

            if p_meas > 0.0:
                syndrome = syndrome ^ gen_err_conf(d * d, p_meas, rng)

            ca.step(xframe, syndrome, p_sig)

            if reset_time is not None and reset_time > 0 and t % reset_time == 0:
                ca.reset()

        lifetimes[i] = t

    total_lifetime = int(np.sum(lifetimes))

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = (
            f"scala2d_phenomenological_d{d}_pdata{p:g}_pmeas{p_meas:g}_psig{p_sig:g}"
            f"_tr{reset_time if reset_time is not None else 0}_seed{seed}"
        ).replace(".", "p")
        np.save(output_dir / f"{stem}.npy", lifetimes)

        meta = {
            "model": "scala2d",
            "noise_model": "phenomenological",
            "observable": "lifetime",
            "distance": d,
            "samples": n_samples,
            "p_data": p,
            "p_meas": p_meas,
            "p_signal": p_sig,
            "reset_time": 0 if reset_time is None else reset_time,
            "seed": seed,
            "mean_lifetime": float(np.mean(lifetimes)),
            "data_file": str(output_dir / f"{stem}.npy"),
        }
        (output_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))

    return total_lifetime, lifetimes
