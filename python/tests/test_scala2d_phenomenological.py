from python.experiments.scala2d_phenomenological import benchmark_pheno


def main() -> None:
    _, low = benchmark_pheno(9, 0.002, 20, seed=12345, p_meas=0.0, p_sig=0.0, reset_time=7)
    _, high = benchmark_pheno(9, 0.01, 20, seed=12345, p_meas=0.0, p_sig=0.0, reset_time=7)

    t_low = low.mean()
    t_high = high.mean()

    print("scala2d pheno low p mean lifetime =", t_low)
    print("scala2d pheno high p mean lifetime =", t_high)

    if not (t_low > t_high):
        raise RuntimeError("Expected mean lifetime to decrease as p_data increases")

    print("test_scala2d_phenomenological passed")


if __name__ == "__main__":
    main()
