#include "models/scala2d_cell.h"

#include <algorithm>

Scala2DCell::Scala2DCell()
{
    reset();
}

void Scala2DCell::reset()
{
    std::fill(defects.begin(), defects.end(), false);
    std::fill(sigs.begin(), sigs.end(), false);
    std::fill(nsigs.begin(), nsigs.end(), false);
}

void Scala2DCell::setNeighbors(std::array<Scala2DCell*, 4>& neighbors_)
{
    neighbors = neighbors_;
}

void Scala2DCell::setDefect(bool value)
{
    defects[Scala2DLocation::C] = value;
}

void Scala2DCell::acquire()
{
    for (int i = 0; i < 4; i++) {
        defects[i] = neighbors[i]->defects[Scala2DLocation::C];
    }
}

void Scala2DCell::broadcast()
{
    // Two independent SCALA1D-like axes.
    if (defects[Scala2DLocation::C]) {
        if (!sigs[Scala2DLocation::W] && !sigs[Scala2DLocation::E]) {
            sigs[Scala2DLocation::W] = true;
            sigs[Scala2DLocation::E] = true;
        }
        if (!sigs[Scala2DLocation::N] && !sigs[Scala2DLocation::S]) {
            sigs[Scala2DLocation::N] = true;
            sigs[Scala2DLocation::S] = true;
        }
    }
}

void Scala2DCell::propagate()
{
    for (int i = 0; i < 4; i++) {
        nsigs[3 - i] = neighbors[i]->sigs[3 - i];
    }
}

void Scala2DCell::update(double psig, std::mt19937& rng)
{
    const int num_nsigs = int(std::accumulate(nsigs.begin(), nsigs.end(), 0.0));

    if (!defects[Scala2DLocation::C] && num_nsigs > 1) {
        for (int i = 0; i < 4; i++) {
            sigs[i] = nsigs[3 - i];  // reflect
        }
    } else {
        for (int i = 0; i < 4; i++) {
            sigs[i] = nsigs[i];      // transmit
        }
    }

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < 4; i++) {
        if (dist(rng) < psig) {
            sigs[i] = sigs[i] ^ 1;
        }
    }
}

Scala2DLocation Scala2DCell::rule()
{
    if (defects[Scala2DLocation::C]) {
        const int num_sigs = int(std::accumulate(sigs.begin(), sigs.end(), 0.0));
        const bool is_isolated_defect = std::accumulate(defects.begin(), defects.end(), 0.0) == 1.0;

        // NN rule with preference.
        if (defects[Scala2DLocation::E]) {
            return Scala2DLocation::E;
        } else if (defects[Scala2DLocation::N]) {
            return Scala2DLocation::N;
        }

        // Follow-field rule.
        if (is_isolated_defect) {
            if (num_sigs == 1) {
                if (sigs[Scala2DLocation::W]) return Scala2DLocation::E;
                if (sigs[Scala2DLocation::E]) return Scala2DLocation::W;
                if (sigs[Scala2DLocation::N]) return Scala2DLocation::S;
                if (sigs[Scala2DLocation::S]) return Scala2DLocation::N;
            } else if (num_sigs == 2) {
                if (sigs[Scala2DLocation::W] && sigs[Scala2DLocation::N]) return Scala2DLocation::S;
                if (sigs[Scala2DLocation::E] && sigs[Scala2DLocation::N]) return Scala2DLocation::W;
            } else if (num_sigs == 3) {
                if (sigs[Scala2DLocation::W] && sigs[Scala2DLocation::N] && sigs[Scala2DLocation::E]) return Scala2DLocation::S;
                if (sigs[Scala2DLocation::S] && sigs[Scala2DLocation::N] && sigs[Scala2DLocation::E]) return Scala2DLocation::W;
                if (sigs[Scala2DLocation::S] && sigs[Scala2DLocation::W] && sigs[Scala2DLocation::E]) return Scala2DLocation::N;
                if (sigs[Scala2DLocation::S] && sigs[Scala2DLocation::W] && sigs[Scala2DLocation::N]) return Scala2DLocation::E;
            }
        }
    }

    return Scala2DLocation::None;
}
