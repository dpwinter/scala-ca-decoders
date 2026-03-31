#ifndef SCALA2D_H
#define SCALA2D_H

#include "models/scala2d_cell.h"
#include <pybind11/numpy.h>

#include <random>
#include <vector>

namespace py = pybind11;

class Scala2D
{
private:
    int L;
    std::vector<Scala2DCell> cells;
    std::mt19937 rng;

public:
    explicit Scala2D(int L, unsigned int seed = 12345);

    void reset();
    void step(py::array_t<bool>& xframe, py::array_t<bool>& syndrome, double psig = 0.0);
};

#endif
