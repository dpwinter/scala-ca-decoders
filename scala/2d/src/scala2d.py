"""
Minimal reference implementation of SCALA2D.

Cell (y, x) is a plaquette of the toric-code lattice.

horizontal[y][x] is the west edge of cell (y, x).
vertical[y][x]   is the north edge of cell (y, x).

Signals are stored as north, west, east, south.
"""


def zeros(d):
    return [[False] * d for _ in range(d)]


def copy(a):
    return [row.copy() for row in a]


def syndrome(horizontal, vertical):
    d = len(horizontal)
    defect = zeros(d)

    for y in range(d):
        for x in range(d):
            defect[y][x] = (
                horizontal[y][x]
                ^ horizontal[y][(x + 1) % d]
                ^ vertical[y][x]
                ^ vertical[(y + 1) % d][x]
            )

    return defect


def reset_signals(d):
    return zeros(d), zeros(d), zeros(d), zeros(d)


def correction(
    defect,
    north_defect,
    west_defect,
    east_defect,
    south_defect,
    north,
    west,
    east,
    south,
):
    if not defect:
        return None

    # Nearest-neighbor corrections
    if west_defect:
        return "W"

    if north_defect:
        return "N"

    # Signal-follow corrections require an isolated defect
    if east_defect or south_defect:
        return None

    nsig = sum((north, west, east, south))

    if nsig == 1:
        if north:
            return "S"
        if west:
            return "E"
        if east:
            return "W"
        if south:
            return "N"

    elif nsig == 2:
        if west and north:
            return "S"
        if east and north:
            return "W"

    elif nsig == 3:
        if west and north and east:
            return "S"
        if south and north and east:
            return "W"
        if south and west and east:
            return "N"
        if south and west and north:
            return "E"

    return None


def step(horizontal, vertical, north, west, east, south):
    d = len(horizontal)

    # Acquire
    defect = syndrome(horizontal, vertical)

    # Broadcast
    broadcast_north = copy(north)
    broadcast_west = copy(west)
    broadcast_east = copy(east)
    broadcast_south = copy(south)

    for y in range(d):
        for x in range(d):
            if defect[y][x]:
                if not west[y][x] and not east[y][x]:
                    broadcast_west[y][x] = True
                    broadcast_east[y][x] = True

                if not north[y][x] and not south[y][x]:
                    broadcast_north[y][x] = True
                    broadcast_south[y][x] = True

    # Propagate
    incoming_north = zeros(d)
    incoming_west = zeros(d)
    incoming_east = zeros(d)
    incoming_south = zeros(d)

    for y in range(d):
        for x in range(d):
            incoming_north[y][x] = broadcast_north[(y + 1) % d][x]
            incoming_west[y][x] = broadcast_west[y][(x + 1) % d]
            incoming_east[y][x] = broadcast_east[y][(x - 1) % d]
            incoming_south[y][x] = broadcast_south[(y - 1) % d][x]

    # Reflect / transmit
    new_north = zeros(d)
    new_west = zeros(d)
    new_east = zeros(d)
    new_south = zeros(d)

    for y in range(d):
        for x in range(d):
            n = incoming_north[y][x]
            w = incoming_west[y][x]
            e = incoming_east[y][x]
            s = incoming_south[y][x]

            if not defect[y][x] and sum((n, w, e, s)) > 1:
                new_north[y][x] = s
                new_west[y][x] = e
                new_east[y][x] = w
                new_south[y][x] = n
            else:
                new_north[y][x] = n
                new_west[y][x] = w
                new_east[y][x] = e
                new_south[y][x] = s

    # Corrections
    flip_horizontal = zeros(d)
    flip_vertical = zeros(d)

    for y in range(d):
        for x in range(d):
            north_defect = defect[(y - 1) % d][x]
            west_defect = defect[y][(x - 1) % d]
            east_defect = defect[y][(x + 1) % d]
            south_defect = defect[(y + 1) % d][x]

            direction = correction(
                defect[y][x],
                north_defect,
                west_defect,
                east_defect,
                south_defect,
                new_north[y][x],
                new_west[y][x],
                new_east[y][x],
                new_south[y][x],
            )

            if direction == "W":
                flip_horizontal[y][x] ^= True

            elif direction == "E":
                flip_horizontal[y][(x + 1) % d] ^= True

            elif direction == "N":
                flip_vertical[y][x] ^= True

            elif direction == "S":
                flip_vertical[(y + 1) % d][x] ^= True

    # Apply corrections synchronously
    new_horizontal = copy(horizontal)
    new_vertical = copy(vertical)

    for y in range(d):
        for x in range(d):
            new_horizontal[y][x] ^= flip_horizontal[y][x]
            new_vertical[y][x] ^= flip_vertical[y][x]

    return (
        new_horizontal,
        new_vertical,
        new_north,
        new_west,
        new_east,
        new_south,
    )


def reset_schedule(d):
    return (
        list(range(1, d + 1))
        + list(range(d - 1, 0, -1))
    )


def decode_code_capacity(horizontal, vertical):
    d = len(horizontal)

    horizontal = copy(horizontal)
    vertical = copy(vertical)

    north, west, east, south = reset_signals(d)

    for reset_time in reset_schedule(d):
        for _ in range(reset_time):
            (
                horizontal,
                vertical,
                north,
                west,
                east,
                south,
            ) = step(
                horizontal,
                vertical,
                north,
                west,
                east,
                south,
            )

        north, west, east, south = reset_signals(d)

    return horizontal, vertical


def is_syndrome_free(horizontal, vertical):
    return not any(
        bit
        for row in syndrome(horizontal, vertical)
        for bit in row
    )
