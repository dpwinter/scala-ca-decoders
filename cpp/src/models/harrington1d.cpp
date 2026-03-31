#include "models/harrington1d.h"

#include <stdexcept>
#include <cassert>
#include <cmath>

// Global CA wrapper:
// - creates lattice of cells
// - wires neighbors (periodic boundary)
// - applies update cycle
Harrington1D::Harrington1D(int L_, int U, double fC, double fN)
    : L{L_}, Q{3}
{
    // Determine number of hierarchy levels
    double df = std::log(static_cast<double>(L)) / std::log(static_cast<double>(Q));
	double rounded = std::round(df);

	if (std::abs(df - rounded) > 1e-10) {
		throw std::invalid_argument("Harrington1D requires distance to be a power of 3.");
	}

	int d = static_cast<int>(rounded);

    cells.reserve(L);

    // Create cells
    for (int i = 0; i < L; i++) {
        cells.emplace_back(i, Q, U, d, fC, fN);
    }

    // Set periodic nearest neighbors
    for (int i = 0; i < L; i++) {
        std::vector<Harrington1DCell*> neighbors(2);

        int l = (i == 0) ? L - 1 : i - 1;
        int r = (i + 1) % L;

        neighbors[H1D_L] = &cells[l];
        neighbors[H1D_R] = &cells[r];

        cells[i].setNeighbors(neighbors);
    }
}

const Harrington1DCell& Harrington1D::getCell(int i) const
{
    assert(i < L);
    return cells[i];
}

// Perform one CA update step:
//
// 1. inject syndrome (defects)
// 2. acquire neighbor information
// 3. update internal memory
// 4. compute local correction directions
std::vector<Harrington1DLocation> Harrington1D::step(const std::vector<bool>& syndrome)
{
    // load defects into cells
    for (int i = 0; i < L; i++) {
        cells[i].setDefect(syndrome[i]);
    }

    // propagate neighbor info
    for (int i = 0; i < L; i++) {
        cells[i].acquire();
    }

    // update internal state (memory + signals)
    for (int i = 0; i < L; i++) {
        cells[i].update();
    }

    // compute correction decisions
    std::vector<Harrington1DLocation> cors;
    cors.reserve(L);

    for (int i = 0; i < L; i++) {
        cors.push_back(cells[i].rule());
    }

    return cors;
}
