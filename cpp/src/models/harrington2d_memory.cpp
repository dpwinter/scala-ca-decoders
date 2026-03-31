#include "models/harrington2d_memory.h"

#include <algorithm>

Harrington2DMemory::Harrington2DMemory(Harrington2DLocation addr_, int U_, int Q_)
    : addr{addr_}, U{U_}, Q{Q_}
{
    age = 0;
    countSig = std::vector<bool>(8, 0);
    n_countSig = std::vector<bool>(8, 0);
    flipSig = std::vector<bool>(4, 0);
    n_flipSig = std::vector<bool>(4, 0);
    count = std::vector<int>(9, 0);
}

void Harrington2DMemory::reset()
{
    std::fill(countSig.begin(), countSig.end(), 0);
    std::fill(n_countSig.begin(), n_countSig.end(), 0);
    std::fill(flipSig.begin(), flipSig.end(), 0);
    std::fill(n_flipSig.begin(), n_flipSig.end(), 0);
    std::fill(count.begin(), count.end(), 0);
}
