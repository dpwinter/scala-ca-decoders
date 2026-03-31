#ifndef HARRINGTON2D_MEMORY_H
#define HARRINGTON2D_MEMORY_H

#include <vector>

enum Harrington2DLocation { N, W, E, S, NW, NE, SW, SE, C, None = -1 };

inline Harrington2DLocation harrington2d_loc_from_coords(int row, int col)
{
    Harrington2DLocation locs[9] = {
        NW, N,  NE,
        W,  C,  E,
        SW, S,  SE
    };
    return locs[3 * row + col];
}

inline Harrington2DLocation harrington2d_opposite_loc(int loc)
{
    Harrington2DLocation locs[8] = {S, E, W, N, SE, SW, NE, NW};
    return locs[loc];
}

struct Harrington2DMemory
{
    Harrington2DLocation addr;
    int U;
    int Q;
    int age;

    std::vector<bool> countSig;
    std::vector<bool> n_countSig;
    std::vector<bool> flipSig;
    std::vector<bool> n_flipSig;
    std::vector<int> count;

    Harrington2DMemory(Harrington2DLocation addr_, int U_, int Q_);
    void reset();
};

#endif
