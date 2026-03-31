#ifndef LOCATION_H_
#define LOCATION_H_

enum Location {N, W, E, S, NW, NE, SW, SE, C, None=-1};
Location locFromCoords(int row, int col);
Location oppositeLoc(int loc);

#endif
