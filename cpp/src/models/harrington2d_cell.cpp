#include "models/harrington2d_cell.h"

#include <cmath>
#include <random>

Harrington2DCell::Harrington2DCell(int row, int col, int Q, int U, int d_)
    : d{d_}
{
    addr = harrington2d_loc_from_coords(row % Q, col % Q);

    for (int k = 1; k < d; k++) {
        int offset = int((std::pow(Q, k) - 1) / 2.0);
        double krow = (row - offset) / std::pow(Q, k);
        double kcol = (col - offset) / std::pow(Q, k);

        Harrington2DLocation kaddr =
            (int(krow) == krow && int(kcol) == kcol)
                ? harrington2d_loc_from_coords(int(krow) % Q, int(kcol) % Q)
                : Harrington2DLocation::None;

        memory.push_back(Harrington2DMemory{kaddr, int(std::pow(U, k)), int(std::pow(Q, k))});
    }

    defects = std::vector<bool>(9, 0);
}

void Harrington2DCell::setNeighbors(std::vector<Harrington2DCell*>& neighbors_)
{
    neighbors = std::move(neighbors_);
}

void Harrington2DCell::setDefect(bool value)
{
    defects[Harrington2DLocation::C] = value;
}

void Harrington2DCell::acquire()
{
    for (int i = 0; i < 8; i++) {
        defects[i] = neighbors[i]->defects[Harrington2DLocation::C];
    }

    for (int k = 0; k < d - 1; k++) {
        for (int i = 0; i < 8; i++) {
            Harrington2DCell opposite_neighbor = *neighbors[harrington2d_opposite_loc(i)];
            memory[k].n_countSig[i] = opposite_neighbor.memory[k].countSig[i];
            if (i < 4) {
                memory[k].n_flipSig[i] = opposite_neighbor.memory[k].flipSig[i];
            }
        }
    }
}

void Harrington2DCell::update(double psig, std::mt19937& rng)
{
	std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (Harrington2DMemory& mem : memory) {
        for (int i = 0; i < 8; i++) {
            if (dist(rng) < psig) {
                mem.n_countSig[i] = mem.n_countSig[i] ^ 1;
            }
        }

        mem.age = (mem.age + 1) % (mem.U + mem.Q + 1);

        if (mem.addr != Harrington2DLocation::None) {
            for (int i = 0; i < 8; i++) {
                mem.countSig[i] = defects[Harrington2DLocation::C];
            }

            mem.count[Harrington2DLocation::C] += defects[Harrington2DLocation::C];
            for (int i = 0; i < 8; i++) {
                mem.count[i] += mem.n_countSig[harrington2d_opposite_loc(i)];
            }
        }

        if (mem.addr == Harrington2DLocation::None) {
            mem.countSig = mem.n_countSig;
            mem.flipSig = mem.n_flipSig;
        }
    }
}

Harrington2DLocation Harrington2DCell::rule()
{
    for (int k = d - 2; k > -1; k--) {
        if (memory[k].age == memory[k].U && memory[k].addr != Harrington2DLocation::None) {
            std::vector<bool> k_defects(9, 0);

            for (int i = 0; i < 9; i++) {
                double f = (i == Harrington2DLocation::C) ? fC : fN;
                k_defects[i] = memory[k].count[i] >= f * memory[k].U;
                memory[k].count[i] = 0;
            }

            Harrington2DLocation dir = harringtonRule(memory[k].addr, k_defects);
            if (dir != Harrington2DLocation::None) {
                memory[k].flipSig[dir] = 1;
            }
        }

        if (memory[k].age == memory[k].U + memory[k].Q) {
            std::vector<bool> temp = memory[k].flipSig;
            memory[k].reset();
            for (int i = 0; i < 4; i++) {
                if (temp[i]) {
                    return Harrington2DLocation(i);
                }
            }
        }
    }

    return harringtonRule(addr, defects);
}

Harrington2DLocation Harrington2DCell::harringtonRule(
    const Harrington2DLocation& addr,
    const std::vector<bool>& defects
)
{
    if (addr == Harrington2DLocation::C || defects[Harrington2DLocation::C] == 0) {
        return Harrington2DLocation::None;
    }

    if (addr == Harrington2DLocation::NW) {
        if (defects[Harrington2DLocation::N]) {
            return Harrington2DLocation::None;
        } else if (defects[Harrington2DLocation::W]) {
            return Harrington2DLocation::W;
        } else if (defects[Harrington2DLocation::E]) {
            return Harrington2DLocation::E;
        } else if (defects[Harrington2DLocation::S]) {
            return Harrington2DLocation::S;
        } else if (defects[Harrington2DLocation::NW]) {
            return Harrington2DLocation::W;
        } else if (defects[Harrington2DLocation::NE]) {
            return Harrington2DLocation::None;
        } else if (defects[Harrington2DLocation::SW]) {
            return Harrington2DLocation::W;
        } else {
            return Harrington2DLocation::E;
        }
    }

    if (addr == Harrington2DLocation::N) {
        if (defects[Harrington2DLocation::N]) {
            return Harrington2DLocation::None;
        } else if (defects[Harrington2DLocation::W]) {
            return Harrington2DLocation::None;
        } else if (defects[Harrington2DLocation::E]) {
            return Harrington2DLocation::None;
        } else if (defects[Harrington2DLocation::S]) {
            return Harrington2DLocation::S;
        } else if (defects[Harrington2DLocation::NW]) {
            return Harrington2DLocation::None;
        } else if (defects[Harrington2DLocation::NE]) {
            return Harrington2DLocation::None;
        } else {
            return Harrington2DLocation::S;
        }
    }

    if (addr == Harrington2DLocation::NE) {
        if (defects[Harrington2DLocation::N]) {
            return Harrington2DLocation::None;
        } else if (defects[Harrington2DLocation::E]) {
            return Harrington2DLocation::None;
        } else if (defects[Harrington2DLocation::W]) {
            return Harrington2DLocation::W;
        } else if (defects[Harrington2DLocation::S]) {
            return Harrington2DLocation::S;
        } else if (defects[Harrington2DLocation::NW]) {
            return Harrington2DLocation::None;
        } else if (defects[Harrington2DLocation::NE]) {
            return Harrington2DLocation::None;
        } else if (defects[Harrington2DLocation::SE]) {
            return Harrington2DLocation::None;
        } else {
            return Harrington2DLocation::W;
        }
    }

    if (addr == Harrington2DLocation::W) {
        if (defects[Harrington2DLocation::W]) {
            return Harrington2DLocation::W;
        } else if (defects[Harrington2DLocation::N]) {
            return Harrington2DLocation::None;
        } else if (defects[Harrington2DLocation::S]) {
            return Harrington2DLocation::None;
        } else if (defects[Harrington2DLocation::E]) {
            return Harrington2DLocation::E;
        } else if (defects[Harrington2DLocation::NW]) {
            return Harrington2DLocation::W;
        } else if (defects[Harrington2DLocation::SW]) {
            return Harrington2DLocation::W;
        } else {
            return Harrington2DLocation::E;
        }
    }

    if (addr == Harrington2DLocation::E) {
        if (defects[Harrington2DLocation::E]) {
            return Harrington2DLocation::None;
        } else if (defects[Harrington2DLocation::N]) {
            return Harrington2DLocation::None;
        } else if (defects[Harrington2DLocation::S]) {
            return Harrington2DLocation::None;
        } else if (defects[Harrington2DLocation::W]) {
            return Harrington2DLocation::W;
        } else if (defects[Harrington2DLocation::NE]) {
            return Harrington2DLocation::None;
        } else if (defects[Harrington2DLocation::SE]) {
            return Harrington2DLocation::None;
        } else {
            return Harrington2DLocation::W;
        }
    }

    if (addr == Harrington2DLocation::SW) {
        if (defects[Harrington2DLocation::W]) {
            return Harrington2DLocation::W;
        } else if (defects[Harrington2DLocation::S]) {
            return Harrington2DLocation::S;
        } else if (defects[Harrington2DLocation::N]) {
            return Harrington2DLocation::N;
        } else if (defects[Harrington2DLocation::E]) {
            return Harrington2DLocation::E;
        } else if (defects[Harrington2DLocation::NW]) {
            return Harrington2DLocation::W;
        } else if (defects[Harrington2DLocation::SW]) {
            return Harrington2DLocation::S;
        } else if (defects[Harrington2DLocation::SE]) {
            return Harrington2DLocation::S;
        } else {
            return Harrington2DLocation::E;
        }
    }

    if (addr == Harrington2DLocation::S) {
        if (defects[Harrington2DLocation::S]) {
            return Harrington2DLocation::S;
        } else if (defects[Harrington2DLocation::W]) {
            return Harrington2DLocation::None;
        } else if (defects[Harrington2DLocation::E]) {
            return Harrington2DLocation::None;
        } else if (defects[Harrington2DLocation::N]) {
            return Harrington2DLocation::N;
        } else if (defects[Harrington2DLocation::SE]) {
            return Harrington2DLocation::S;
        } else if (defects[Harrington2DLocation::SW]) {
            return Harrington2DLocation::S;
        } else {
            return Harrington2DLocation::N;
        }
    }

    if (addr == Harrington2DLocation::SE) {
        if (defects[Harrington2DLocation::E]) {
            return Harrington2DLocation::None;
        } else if (defects[Harrington2DLocation::S]) {
            return Harrington2DLocation::S;
        } else if (defects[Harrington2DLocation::W]) {
            return Harrington2DLocation::W;
        } else if (defects[Harrington2DLocation::N]) {
            return Harrington2DLocation::N;
        } else if (defects[Harrington2DLocation::NE]) {
            return Harrington2DLocation::None;
        } else if (defects[Harrington2DLocation::SE]) {
            return Harrington2DLocation::None;
        } else if (defects[Harrington2DLocation::SW]) {
            return Harrington2DLocation::S;
        } else {
            return Harrington2DLocation::W;
        }
    }

    return Harrington2DLocation::None;
}
