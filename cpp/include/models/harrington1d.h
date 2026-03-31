#ifndef HARRINGTON1D_H
#define HARRINGTON1D_H

#include "models/harrington1d_cell.h"

#include <vector>

class Harrington1D
{
private:
    int L;
    int Q;
    std::vector<Harrington1DCell> cells;

public:
    Harrington1D(int L_, int U, double fC, double fN);

    const Harrington1DCell& getCell(int i) const;

    std::vector<Harrington1DLocation> step(const std::vector<bool>& syndrome);
};

#endif
