#include "codes/repetition_code.h"

#include <iostream>
#include <random>
#include <vector>

namespace {

bool all_zero(const std::vector<bool>& x)
{
    for (bool v : x) {
        if (v) {
            return false;
        }
    }
    return true;
}

int count_true(const std::vector<bool>& x)
{
    int c = 0;
    for (bool v : x) {
        c += static_cast<int>(v);
    }
    return c;
}

}  // namespace

int main()
{
    RepetitionCode code(9);

    // Fresh code should have no errors and no syndrome.
    if (code.num_errors() != 0) {
        std::cerr << "Fresh code has nonzero error count\n";
        return 1;
    }

    auto syn = code.syndrome();
    if (!all_zero(syn)) {
        std::cerr << "Fresh code has nonzero syndrome\n";
        return 1;
    }

    // A single qubit flip should create exactly two defects.
    code.flip_qubit(3);
    syn = code.syndrome();

    if (code.num_errors() != 1) {
        std::cerr << "Single flip did not produce exactly one data error\n";
        return 1;
    }

    if (count_true(syn) != 2) {
        std::cerr << "Single flip did not produce exactly two defects\n";
        return 1;
    }

    if (code.has_logical_error()) {
        std::cerr << "Single flip should not cause a logical error\n";
        return 1;
    }

    // More than half the qubits flipped should be a logical error.
    code.reset();
    for (int i = 0; i < 5; ++i) {
        code.flip_qubit(i);
    }

    if (!code.has_logical_error()) {
        std::cerr << "Five flips on distance-9 code should be a logical error\n";
        return 1;
    }

    std::cout << "test_repetition_code passed\n";
    return 0;
}
