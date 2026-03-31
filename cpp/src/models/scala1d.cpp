#include "models/scala1d.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

Scala1D::Scala1D(std::size_t distance)
    : ncells_(distance), cells_(distance, 0)
{
    if (distance == 0) {
        throw std::invalid_argument("Scala1D distance must be positive.");
    }
}

void Scala1D::reset()
{
    std::fill(cells_.begin(), cells_.end(), 0);
}

std::size_t Scala1D::distance() const noexcept
{
    return ncells_;
}

void Scala1D::step(
    RepetitionCode& code,
    const std::vector<bool>& syndrome,
    double p_signal,
    std::mt19937& rng
)
{
    if (syndrome.size() != ncells_) {
        throw std::invalid_argument("Scala1D::step: syndrome size does not match decoder size.");
    }
    if (code.distance() != ncells_) {
        throw std::invalid_argument("Scala1D::step: code size does not match decoder size.");
    }
    if (p_signal < 0.0 || p_signal > 1.0) {
        throw std::invalid_argument("Scala1D::step: p_signal must be in [0,1].");
    }

    std::vector<int> next_cells(ncells_, 0);

    for (std::size_t i = 0; i < ncells_; ++i) {
        const std::size_t i_prev_prev = mod(static_cast<long long>(i) - 2);
        const std::size_t i_prev = mod(static_cast<long long>(i) - 1);
        const std::size_t i_next = mod(static_cast<long long>(i) + 1);

        // Local syndrome neighborhood. (due to syndrome convention)
        const bool sl = syndrome[i_prev_prev];
        const bool sc = syndrome[i_prev];
        const bool sr = syndrome[i];
        // const bool sl = syndrome[i_prev];
        // const bool sc = syndrome[i];
        // const bool sr = syndrome[i_next];

        // Current CA states in the neighborhood.
        const int stl = cells_[i_prev];
        const int stc = cells_[i];
        const int str = cells_[i_next];
        (void)stc; // not used in update

        // Incoming left/right signals.
        bool li = (str & 2) || (!(str & 1) && !(str & 2) && sr);
        bool ri = (stl & 1) || (!(stl & 1) && !(stl & 2) && sl);

        // Greedy local correction rule.
        if (sl && sc) {
            code.flip_qubit(i_prev);
        }
        else if (!sl && sc && !sr && !li && ri) {
            code.flip_qubit(i_prev);
        }
        else if (!sr && sc && !sl && li && !ri) {
            code.flip_qubit(i);
        }

        // Optional signal noise.
        if (sample_bernoulli(p_signal, rng)) {
            li = !li;
        }
        if (sample_bernoulli(p_signal, rng)) {
            ri = !ri;
        }

        // Encode state as: 0 = none, 1 = right, 2 = left, 3 = both.
        next_cells[i] = 2 * static_cast<int>(li) + static_cast<int>(ri);
    }

    cells_ = std::move(next_cells);
}

void Scala1D::step(
    RepetitionCode& code,
    double p_signal,
    std::mt19937& rng
)
{
    step(code, code.syndrome(), p_signal, rng);
}

std::ostream& operator<<(std::ostream& out, const Scala1D& ca)
{
    for (std::size_t i = 0; i < ca.ncells_; ++i) {
        switch (ca.cells_[i]) {
            case 0:
                out << "  ";
                break;
            case 1:
                out << "> ";
                break;
            case 2:
                out << "< ";
                break;
            case 3:
                out << "\u2277 ";
                break;
            default:
                out << "? ";
                break;
        }
    }
    out << '\n';
    return out;
}

std::size_t Scala1D::mod(long long idx) const noexcept
{
    const long long L = static_cast<long long>(ncells_);
    return static_cast<std::size_t>((idx % L + L) % L);
}

bool Scala1D::sample_bernoulli(double p, std::mt19937& rng)
{
    std::bernoulli_distribution dist(p);
    return dist(rng);
}
