#include "models/harrington1d.h"

#include <iostream>
#include <stdexcept>

int main()
{
    try {
        Harrington1D ok(9, 16, 0.9, 0.4);
    } catch (...) {
        std::cerr << "Unexpected failure for valid distance 9\n";
        return 1;
    }

    try {
        Harrington1D bad(15, 16, 0.9, 0.4);
        std::cerr << "Expected failure for invalid distance 15\n";
        return 1;
    } catch (const std::invalid_argument&) {
        // expected
    } catch (...) {
        std::cerr << "Wrong exception type for invalid distance 15\n";
        return 1;
    }

    std::cout << "test_harrington1d_constructor passed\n";
    return 0;
}
