#ifndef SCALA1D_H
#define SCALA1D_H

#include "codes/repetition_code.h"

#include <cstddef>
#include <iosfwd>
#include <random>
#include <vector>

class Scala1D
{
public:
    explicit Scala1D(std::size_t distance);

    void reset();

    std::size_t distance() const noexcept;

    // Perform one CA step using an externally provided syndrome.
    void step(
        RepetitionCode& code,
        const std::vector<bool>& syndrome,
        double p_signal,
        std::mt19937& rng
    );

    // Convenience overload using the ideal syndrome.
    void step(
        RepetitionCode& code,
        double p_signal,
        std::mt19937& rng
    );

    friend std::ostream& operator<<(std::ostream& out, const Scala1D& ca);

private:
    std::size_t mod(long long idx) const noexcept;
    static bool sample_bernoulli(double p, std::mt19937& rng);

    std::size_t ncells_;
    std::vector<int> cells_;
};

#endif
