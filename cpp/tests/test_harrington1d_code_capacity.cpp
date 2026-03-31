#include "experiments/harrington1d_code_capacity.h"
#include "experiments/outcome_utils.h"

#include <iostream>

int main()
{
    const int samples = 3000;
    const int max_steps = 1000;
    const double p_data = 0.15;
    const int U = 16;
    const double fN = 0.4;
    const double fC = 0.9;
    const unsigned int seed = 12345;

    const auto outcomes_d9 = run_harrington1d_code_capacity(
        9,
        samples,
        max_steps,
        p_data,
        U,
        fN,
        fC,
        seed
    );

    const auto outcomes_d27 = run_harrington1d_code_capacity(
        27,
        samples,
        max_steps,
        p_data,
        U,
        fN,
        fC,
        seed
    );

    const double p9 = logical_error_rate(outcomes_d9);
    const double p27 = logical_error_rate(outcomes_d27);

    std::cout << "harrington1d code-capacity pL(d=9)  = " << p9 << "\n";
    std::cout << "harrington1d code-capacity pL(d=27) = " << p27 << "\n";

    if (!(p27 < p9)) {
        std::cerr << "Expected distance-27 to outperform distance-9\n";
        return 1;
    }

    std::cout << "test_harrington1d_code_capacity passed\n";
    return 0;
}
