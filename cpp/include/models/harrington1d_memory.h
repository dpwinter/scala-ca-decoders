#ifndef HARRINGTON1D_MEMORY_H
#define HARRINGTON1D_MEMORY_H

#include <vector>

enum Harrington1DLocation { H1D_L, H1D_R, H1D_C, H1D_None = -1 };

inline Harrington1DLocation harrington1d_loc_from_coords(int addr)
{
    Harrington1DLocation locs[3] = {H1D_L, H1D_C, H1D_R};
    return locs[addr];
}

inline Harrington1DLocation harrington1d_opposite_loc(int loc)
{
    Harrington1DLocation locs[2] = {H1D_R, H1D_L};
    return locs[loc];
}

struct Harrington1DMemory
{
    Harrington1DLocation addr;
    int U;
    int Q;
    int age;

    std::vector<bool> countSig;
    std::vector<bool> n_countSig;
    std::vector<bool> flipSig;
    std::vector<bool> n_flipSig;
    std::vector<int> count;

    Harrington1DMemory(Harrington1DLocation addr_, int U_, int Q_);
    void reset();
};

#endif
