#include "models/scala2d.h"

Scala2D::Scala2D(int L_, unsigned int seed) : L{L_}, rng(seed)
{
    cells.resize(L * L);

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            std::array<Scala2DCell*, 4> neighbors{
                &cells[((i - 1 + L) % L) * L + j],   // N
                &cells[i * L + ((j - 1 + L) % L)],   // W
                &cells[i * L + ((j + 1) % L)],       // E
                &cells[((i + 1) % L) * L + j]        // S
            };
            cells[i * L + j].setNeighbors(neighbors);
        }
    }
}

void Scala2D::reset()
{
    for (auto& cell : cells) {
        cell.reset();
    }
}

void Scala2D::step(py::array_t<bool>& xframe, py::array_t<bool>& syndrome, double psig)
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
        cells[i].broadcast();
    }

    for (int i = 0; i < L * L; i++) {
        cells[i].propagate();
    }

    for (int i = 0; i < L * L; i++) {
        cells[i].update(psig, rng);
    }

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            Scala2DLocation cor = cells[i * L + j].rule();
            int idx = 0;

            switch (cor) {
                case Scala2DLocation::N:
                    idx = L * L + L * (((i - 1) % L + L) % L) + j;
                    break;
                case Scala2DLocation::W:
                    idx = i * L + j;
                    break;
                case Scala2DLocation::E:
                    idx = i * L + (j + 1) % L;
                    break;
                case Scala2DLocation::S:
                    idx = L * L + i * L + j;
                    break;
                default:
                    break;
            }

            if (cor != Scala2DLocation::None) {
                xframe_ptr[idx] = !xframe_ptr[idx];
            }
        }
    }
}
