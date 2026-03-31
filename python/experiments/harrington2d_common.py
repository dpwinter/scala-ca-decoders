from __future__ import annotations

import numpy as np
from scipy.sparse import block_diag, csc_matrix, eye, hstack, kron


def repetition_code(n: int) -> csc_matrix:
    """
    Parity-check matrix of a repetition code with length n.
    """
    row_ind, col_ind = zip(*((i, j) for i in range(n) for j in (i, (i + 1) % n)))
    data = np.ones(2 * n, dtype=np.uint8)
    return csc_matrix((data, (row_ind, col_ind)))


def toric_code_z_stabilisers(L: int) -> csc_matrix:
    """
    Sparse check matrix for the X stabilisers of a toric code with lattice size L.
    Matches the legacy construction.
    """
    Hr = repetition_code(L)
    H = hstack(
        [kron(eye(Hr.shape[1]), Hr), kron(Hr.T, eye(Hr.shape[0]))],
        dtype=np.uint8,
    )
    H.data = H.data % 2
    H.eliminate_zeros()
    return csc_matrix(H)


def toric_code_z_logicals(L: int) -> csc_matrix:
    """
    Sparse binary matrix with each row corresponding to an X logical operator
    of a toric code with lattice size L. Matches the legacy construction.
    """
    H1 = csc_matrix(([1], ([0], [0])), shape=(1, L), dtype=np.uint8)
    H0 = csc_matrix(np.ones((1, L), dtype=np.uint8))
    x_logicals = block_diag([kron(H0, H1), kron(H1, H0)])
    x_logicals.data = x_logicals.data % 2
    x_logicals.eliminate_zeros()
    return csc_matrix(x_logicals)


def gen_err_conf(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(n) < p).astype(bool)
