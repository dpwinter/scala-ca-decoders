#ifndef HARRINGTON2D_H
#define HARRINGTON2D_H

#include "models/harrington2d_cell.h"
#include <pybind11/numpy.h>

#include <random>
#include <vector>

namespace py = pybind11;

class Harrington2D
{
private:
    int L;
    int Q;
    std::vector<Harrington2DCell> cells;
    std::mt19937 rng;

public:
    explicit Harrington2D(int L, unsigned int seed = 12345);

    void step(py::array_t<bool>& xframe, py::array_t<bool>& syndrome, double psig = 0.0);
};

#endif
