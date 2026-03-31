#include "models/harrington1d_cell.h"

#include <cmath>

// Each cell corresponds to one physical qubit position.
// It maintains a hierarchy of "memory levels" that aggregate defect signals
// over increasing spatial/temporal scales.
Harrington1DCell::Harrington1DCell(int addr_, int Q, int U, int d_, double fC_, double fN_)
    : d{d_}, fN{fN_}, fC{fC_}
{
    // Local role of this cell within its Q=3 block (L, C, R)
    addr = harrington1d_loc_from_coords(addr_ % Q);

    // Build hierarchical memory levels k = 1,...,d-1
    // Each level aggregates over a region of size Q^k and time U^k
    for (int k = 1; k < d; k++) {
        int offset = int((std::pow(Q, k) - 1) / 2.0);

        // Determine whether this cell is a "center" at level k
        double addr__ = (addr_ - offset) / std::pow(Q, k);

        Harrington1DLocation kaddr =
            (int(addr__) == addr__)
                ? harrington1d_loc_from_coords(int(addr__) % Q)
                : H1D_None;

        memory.push_back(Harrington1DMemory{
            kaddr,
            int(std::pow(U, k)),   // time window
            int(std::pow(Q, k))    // spatial window
        });
    }

    // defects[L], defects[C], defects[R]
    defects = std::vector<bool>(3, 0);
}

// Set nearest neighbors (periodic boundary handled outside)
void Harrington1DCell::setNeighbors(std::vector<Harrington1DCell*>& neighbors_)
{
    neighbors = std::move(neighbors_);
}

// Set local defect (syndrome bit at this site)
void Harrington1DCell::setDefect(bool val)
{
    defects[H1D_C] = val;
}

// Acquire neighbor information:
// - copy neighboring defects
// - propagate incoming signals from previous timestep
void Harrington1DCell::acquire()
{
    // nearest-neighbor defects
    defects[H1D_L] = neighbors[H1D_L]->defects[H1D_C];
    defects[H1D_R] = neighbors[H1D_R]->defects[H1D_C];

    // propagate signals for each memory level
    for (int k = 0; k < d - 1; k++) {
        // note the crossing: left comes from right neighbor and vice versa
        memory[k].n_countSig[H1D_L] = neighbors[H1D_R]->memory[k].countSig[H1D_L];
        memory[k].n_countSig[H1D_R] = neighbors[H1D_L]->memory[k].countSig[H1D_R];

        memory[k].n_flipSig[H1D_L] = neighbors[H1D_R]->memory[k].flipSig[H1D_L];
        memory[k].n_flipSig[H1D_R] = neighbors[H1D_L]->memory[k].flipSig[H1D_R];
    }
}

// Update internal memory state:
// - advance time (age)
// - accumulate counts
// - propagate signals or inject new ones depending on role
void Harrington1DCell::update()
{
    for (Harrington1DMemory& mem : memory) {
        // periodic clock for this level
        mem.age = (mem.age + 1) % (mem.U + mem.Q);

        // If this cell is a "center" at this level,
        // it generates fresh signals and accumulates statistics
        if (mem.addr != H1D_None) {
            // emit count signals based on current defect
            mem.countSig[H1D_L] = defects[H1D_C];
            mem.countSig[H1D_R] = defects[H1D_C];

            // accumulate counts over time window
            mem.count[H1D_C] += defects[H1D_C];
            mem.count[H1D_L] += mem.n_countSig[H1D_R];
            mem.count[H1D_R] += mem.n_countSig[H1D_L];
        }

        // If not a center, just propagate incoming signals
        if (mem.addr == H1D_None) {
            mem.countSig = mem.n_countSig;
            mem.flipSig = mem.n_flipSig;
        }
    }
}

// Compute correction direction for this cell
Harrington1DLocation Harrington1DCell::rule()
{
    // Traverse hierarchy top-down (large → small scale)
    for (int k = d - 2; k > -1; k--) {

        // At specific time: evaluate whether enough evidence accumulated
        if (memory[k].age == memory[k].U - 1 && memory[k].addr != H1D_None) {
            std::vector<bool> k_defects(3, 0);

            // threshold decisions
            k_defects[H1D_C] = memory[k].count[H1D_C] >= fC * memory[k].U;
            k_defects[H1D_L] = memory[k].count[H1D_L] >= fN * memory[k].U;
            k_defects[H1D_R] = memory[k].count[H1D_R] >= fN * memory[k].U;

            // apply Harrington rule at this level
            Harrington1DLocation dir = harringtonRule(memory[k].addr, k_defects);
            if (dir != H1D_None) {
                memory[k].flipSig[dir] = 1;
            }
        }

        // Later in the cycle: release accumulated flip signals
        if (memory[k].age == memory[k].U + memory[k].Q - 1) {
            std::vector<bool> temp = memory[k].flipSig;
            memory[k].reset();

            // return first active direction
            for (int i = 0; i < 2; i++) {
                if (temp[i]) {
                    return Harrington1DLocation(i);
                }
            }
        }
    }

    // fallback: local rule on immediate neighborhood
    return harringtonRule(addr, defects);
}

// Local Harrington rule:
// determines correction direction based on defect configuration
Harrington1DLocation Harrington1DCell::harringtonRule(
    const Harrington1DLocation& addr_,
    const std::vector<bool>& defects_
)
{
    // center or no defect → no action
    if (addr_ == H1D_C || defects_[H1D_C] == 0) {
        return H1D_None;
    }

    if (addr_ == H1D_L) {
        return defects_[H1D_L] ? H1D_L : H1D_R;
    }

    if (addr_ == H1D_R) {
        return defects_[H1D_R] ? H1D_None : H1D_L;
    }

    return H1D_None;
}
