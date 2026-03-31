#ifndef SCALA2D_CELL_H
#define SCALA2D_CELL_H

#include <array>
#include <numeric>
#include <random>

enum Scala2DLocation { N, W, E, S, C, None = -1 };

class Scala2DCell
{
private:
    std::array<Scala2DCell*, 4> neighbors{};
    std::array<bool, 4> sigs{};
    std::array<bool, 4> nsigs{};
    std::array<bool, 5> defects{};

public:
    Scala2DCell();

    void setNeighbors(std::array<Scala2DCell*, 4>& neighbors_);
    void setDefect(bool value);
    void reset();

    void acquire();
    void broadcast();
    void propagate();
    void update(double psig, std::mt19937& rng);
    Scala2DLocation rule();
};

#endif
