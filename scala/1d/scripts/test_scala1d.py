from itertools import product

from src.scala1d import syndrome, step


def decode(errors):
    d = len(errors)

    left = [False] * d
    right = [False] * d

    for _ in range(d - 2):
        defect = syndrome(errors)
        errors, left, right = step(errors, defect, left, right)

    return errors


def test_ml_equivalence():
    for d in [3, 5, 7, 9]:

        for errors in product([False, True], repeat=d):
            errors = list(errors)

            decoded = decode(errors)

            logical = sum(errors) > d // 2
            expected = [logical] * d

            assert decoded == expected

        print(f"d={d}: all {2**d} configurations passed")


if __name__ == "__main__":
    test_ml_equivalence()
