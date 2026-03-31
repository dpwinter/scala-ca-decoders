from __future__ import annotations

import argparse

from python.experiments.harrington2d_code_capacity import benchmark_code_capacity
from python.experiments.harrington2d_phenomenological import benchmark_pheno


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-model", choices=["code_capacity", "phenomenological"], required=True)
    parser.add_argument("--distance", type=int, required=True)
    parser.add_argument("--p-data", type=float, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--p-meas", type=float, default=0.0)
    parser.add_argument("--p-signal", type=float, default=0.0)
    parser.add_argument("--reset-time", type=int, default=0)

    args = parser.parse_args()

    if args.noise_model == "code_capacity":
        n_fail, outcomes = benchmark_code_capacity(
            args.distance,
            args.p_data,
            args.samples,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        print("model harrington2d")
        print("noise_model code_capacity")
        print(f"logical_error_rate {outcomes.mean()}")
        print(f"num_samples {len(outcomes)}")
        print(f"n_fail {n_fail}")
        return

    if args.reset_time > args.distance:
        raise ValueError("harrington2d phenomenological requires --reset-time <= --distance")

    total_lifetime, lifetimes = benchmark_pheno(
        args.distance,
        args.p_data,
        args.samples,
        seed=args.seed,
        output_dir=args.output_dir,
        p_meas=args.p_meas,
        p_sig=args.p_signal,
        reset_time=(None if args.reset_time == 0 else args.reset_time),
    )
    print("model harrington2d")
    print("noise_model phenomenological")
    print(f"mean_lifetime {lifetimes.mean()}")
    print(f"num_samples {len(lifetimes)}")
    print(f"total_lifetime {total_lifetime}")


if __name__ == "__main__":
    main()
