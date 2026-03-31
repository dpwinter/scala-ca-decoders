#ifndef HARRINGTON2D_CELL_H
#define HARRINGTON2D_CELL_H

#include "models/harrington2d_memory.h"

#include <random>
#include <vector>

class Harrington2DCell
{
private:
    Harrington2DLocation addr;
    int d;
    double fN = 4.0 / 10.0;
    double fC = 9.0 / 10.0;

    std::vector<Harrington2DCell*> neighbors;
    std::vector<bool> defects;
    std::vector<Harrington2DMemory> memory;

public:
    Harrington2DCell(int row, int col, int Q, int U, int d);

    void setNeighbors(std::vector<Harrington2DCell*>& neighbors_);
    void setDefect(bool value);

    void acquire();
    void update(double psig, std::mt19937& rng);
    Harrington2DLocation rule();

    Harrington2DLocation harringtonRule(
        const Harrington2DLocation& addr,
        const std::vector<bool>& defects
    );
};

#endif
