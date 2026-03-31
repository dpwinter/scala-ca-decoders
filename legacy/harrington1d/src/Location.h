#ifndef LOCATION_H_
#define LOCATION_H_

enum Location {L, R, C, None=-1};
inline Location locFromCoords(int addr) {
	Location locs[3] = {Location::L, Location::C, Location::R};
	return locs[addr];
}
inline Location oppositeLoc(int loc) {
	Location locs[2] = {Location::R, Location::L};
	return locs[loc];
}

#endif
