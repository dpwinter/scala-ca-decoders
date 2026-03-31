#include "models/harrington2d.h"

#include <cmath>
#include <stdexcept>

Harrington2D::Harrington2D(int L_, unsigned int seed) : L{L_}, Q{3}, rng(seed)
{
    double df = std::log(static_cast<double>(L)) / std::log(static_cast<double>(Q));
    double rounded = std::round(df);

    if (std::abs(df - rounded) > 1e-10) {
        throw std::invalid_argument("Harrington2D requires distance to be a power of 3.");
    }

    int d = static_cast<int>(rounded);

    cells.reserve(L * L);

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            cells.emplace_back(i, j, Q, 16, d);
        }
    }

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            std::vector<Harrington2DCell*> neighbors;

            int i_ = (i == 0) ? L - 1 : i - 1;
            int j_ = (j == 0) ? L - 1 : j - 1;
            int ip = (i + 1) % L;
            int jp = (j + 1) % L;

            neighbors.push_back(&cells[i_ * L + j]);   // N
            neighbors.push_back(&cells[i * L + j_]);   // W
            neighbors.push_back(&cells[i * L + jp]);   // E
            neighbors.push_back(&cells[ip * L + j]);   // S
            neighbors.push_back(&cells[i_ * L + j_]);  // NW
            neighbors.push_back(&cells[i_ * L + jp]);  // NE
            neighbors.push_back(&cells[ip * L + j_]);  // SW
            neighbors.push_back(&cells[ip * L + jp]);  // SE

            cells[i * L + j].setNeighbors(neighbors);
        }
    }
}

void Harrington2D::step(py::array_t<bool>& xframe, py::array_t<bool>& syndrome, double psig)
{
    auto syndrome_buf = syndrome.request(true);
    bool* syndrome_ptr = static_cast<bool*>(syndrome_buf.ptr);

    auto xframe_buf = xframe.request(true);
    bool* xframe_ptr = static_cast<bool*>(xframe_buf.ptr);

    for (int i = 0; i < L * L; i++) {
        cells[i].setDefect(syndrome_ptr[i]);
    }

    for (int i = 0; i < L * L; i++) {
        cells[i].acquire();
    }

    for (int i = 0; i < L * L; i++) {
        cells[i].update(psig, rng);
    }

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            Harrington2DLocation cor = cells[i * L + j].rule();
            int idx = 0;

            switch (cor) {
                case Harrington2DLocation::N:
                    idx = L * L + L * (((i - 1) % L + L) % L) + j;
                    break;
                case Harrington2DLocation::W:
                    idx = i * L + j;
                    break;
                case Harrington2DLocation::E:
                    idx = i * L + (j + 1) % L;
                    break;
                case Harrington2DLocation::S:
                    idx = L * L + i * L + j;
                    break;
                default:
                    break;
            }

            if (cor != Harrington2DLocation::None) {
                xframe_ptr[idx] = !xframe_ptr[idx];
            }
        }
    }
}
