from python.experiments.scala2d_code_capacity import benchmark_code_capacity


def main() -> None:
    _, out_d9 = benchmark_code_capacity(9, 0.06, 50, seed=12345)
    _, out_d27 = benchmark_code_capacity(27, 0.06, 50, seed=12345)

    p9 = out_d9.mean()
    p27 = out_d27.mean()

    print("scala2d cc pL(d=9)  =", p9)
    print("scala2d cc pL(d=27) =", p27)

    if not (p27 < p9):
        raise RuntimeError("Expected distance-27 to outperform distance-9")

    print("test_scala2d_distance_trend passed")


if __name__ == "__main__":
    main()
