#include "Location.h"

Location locFromCoords(int row, int col) {
    Location locs[9] = {Location::NW, Location::N, Location::NE,
                        Location::W,  Location::C, Location::E,
                        Location::SW, Location::S, Location::SE};
    return locs[3*row + col];
}

Location oppositeLoc(int loc) {
    Location locs[8] = {Location::S, Location::E, Location::W, Location::N,
                        Location::SE, Location::SW, Location::NE, Location::NW};
    return locs[loc];
}
