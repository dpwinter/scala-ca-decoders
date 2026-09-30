"""
Minimal reference implementation of SCALA1D.

Convention
----------
Cell i sits between qubits i-1 and i.

The measured defect at cell i is therefore the parity

    defect[i] = errors[i-1] XOR errors[i]

before measurement noise is applied.

Each cell stores two signal bits:

    left[i]   : signal travelling to the left
    right[i]  : signal travelling to the right

One synchronous SCALA1D update consists of:

    1. Acquire measured defects
    2. Broadcast
    3. Propagate
    4. Correct

Noise generation, reset schedules, logical-failure checks,
Monte Carlo sampling, and data storage belong in the run scripts.
"""


def syndrome(errors):
    """Return the perfect repetition-code syndrome."""
    d = len(errors)
    defect = [False] * d

    for i in range(d):
        defect[i] = errors[(i - 1) % d] != errors[i]

    return defect


def step(errors, defect, left, right):
    """
    Apply one synchronous SCALA1D update.

    Parameters
    ----------
    errors : list[bool]
        Current data-qubit error configuration.

    defect : list[bool]
        Measured syndrome supplied to the CA.
        This may already contain measurement errors.

    left, right : list[bool]
        Current left- and right-moving signal bits.

    Returns
    -------
    new_errors, new_left, new_right
    """
    d = len(errors)

    # ------------------------------------------------------------
    # 1. Acquire
    #
    # `defect` is the measured syndrome supplied to the CA.
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # 2. Broadcast
    #
    # A defect broadcasts in both directions iff both of its
    # signal bits are currently empty.
    # ------------------------------------------------------------

    broadcast_left = left.copy()
    broadcast_right = right.copy()

    for i in range(d):
        if defect[i] and not left[i] and not right[i]:
            broadcast_left[i] = True
            broadcast_right[i] = True

    # ------------------------------------------------------------
    # 3. Propagate
    #
    # Every signal moves by one cell in its direction.
    # ------------------------------------------------------------

    new_left = [False] * d
    new_right = [False] * d

    for i in range(d):
        new_left[i] = broadcast_left[(i + 1) % d]
        new_right[i] = broadcast_right[(i - 1) % d]

    # ------------------------------------------------------------
    # 4. Correct
    #
    # Each cell can flip at most one adjacent qubit per round.
    # The local conditions are mutually exclusive for neighboring
    # cells, so no data qubit is corrected more than once per round.
    # ------------------------------------------------------------

    flip = [False] * d

    for i in range(d):
        L = (i - 1) % d
        R = (i + 1) % d

        # Nearest-neighbor rule:
        # neighboring defects annihilate across their shared qubit.
        if defect[L] and defect[i]:
            flip[L] = True

        # Signal-follow rule:
        # only isolated defects respond to signals.
        elif not defect[L] and defect[i] and not defect[R]:

            # Right-moving signal arrived from the left:
            # move the defect left.
            if new_right[i] and not new_left[i]:
                flip[L] = True

            # Left-moving signal arrived from the right:
            # move the defect right.
            elif new_left[i] and not new_right[i]:
                flip[i] = True

    new_errors = errors.copy()

    for i in range(d):
        if flip[i]:
            new_errors[i] = not new_errors[i]

    return new_errors, new_left, new_right


def reset_signals(left, right):
    """Reset all signal bits."""
    d = len(left)

    new_left = [False] * d
    new_right = [False] * d

    return new_left, new_right
