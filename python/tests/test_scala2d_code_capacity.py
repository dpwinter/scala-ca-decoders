from python.experiments.scala2d_code_capacity import benchmark_code_capacity


def main() -> None:
    _, low = benchmark_code_capacity(9, 0.02, 50, seed=12345)
    _, high = benchmark_code_capacity(9, 0.08, 50, seed=12345)

    p_low = low.mean()
    p_high = high.mean()

    print("scala2d cc p_low =", p_low)
    print("scala2d cc p_high =", p_high)

    if not (p_low < p_high):
        raise RuntimeError("Expected logical error rate to increase with p_data")

    print("test_scala2d_code_capacity passed")


if __name__ == "__main__":
    main()
