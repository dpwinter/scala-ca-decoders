#ifndef HARRINGTON1D_CELL_H
#define HARRINGTON1D_CELL_H

#include "models/harrington1d_memory.h"

#include <vector>

class Harrington1DCell
{
private:
    Harrington1DLocation addr;
    int d;
    double fN;
    double fC;

    std::vector<Harrington1DCell*> neighbors;
    std::vector<bool> defects;
    std::vector<Harrington1DMemory> memory;

public:
    Harrington1DCell(int addr_, int Q, int U, int d_, double fC_, double fN_);

    void setNeighbors(std::vector<Harrington1DCell*>& neighbors_);
    void setDefect(bool value);

    void acquire();
    void update();
    Harrington1DLocation rule();

    Harrington1DLocation harringtonRule(
        const Harrington1DLocation& addr_,
        const std::vector<bool>& defects_
    );
};

#endif
