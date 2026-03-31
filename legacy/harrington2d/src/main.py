import harrington
import pymatching as pm
from pymatching import Matching
from scipy.sparse import hstack, kron, eye, csc_matrix, block_diag
import numpy as np
import csv
from pathlib import Path

def repetition_code(n):
    """
    Parity check matrix of a repetition code with length n.
    """
    row_ind, col_ind = zip(*((i, j) for i in range(n) for j in (i, (i+1)%n)))
    data = np.ones(2*n, dtype=np.uint8)
    return csc_matrix((data, (row_ind, col_ind)))

def toric_code_z_stabilisers(L):
    """
    Sparse check matrix for the X stabilisers of a toric code with
    lattice size L, constructed as the hypergraph product of
    two repetition codes.
    """
    Hr = repetition_code(L)
    H = hstack(
            [kron(eye(Hr.shape[1]), Hr), kron(Hr.T, eye(Hr.shape[0]))],
            dtype=np.uint8
        )
    H.data = H.data % 2
    H.eliminate_zeros()
    return csc_matrix(H)

def toric_code_z_logicals(L):
    """
    Sparse binary matrix with each row corresponding to an X logical operator
    of a toric code with lattice size L. Constructed from the
    homology groups of the repetition codes using the Kunneth
    theorem.
    """
    H1 = csc_matrix(([1], ([0],[0])), shape=(1,L), dtype=np.uint8)
    H0 = csc_matrix(np.ones((1, L), dtype=np.uint8))
    x_logicals = block_diag([kron(H0, H1), kron(H1, H0)])
    x_logicals.data = x_logicals.data % 2
    x_logicals.eliminate_zeros()
    return csc_matrix(x_logicals)

def print_toric_code(d, xframe, syndrome):
    # xframe: 0...d^2 (=hgrid) , d^2+1...2d^2 (=vgrid)

    # h_grid = xframe[:d*d].reshape(d,d).astype(int)
    # v_grid = xframe[d*d:].reshape(d,d).astype(int)
    # syn_grid = syndrome.reshape(d,d).astype(int)

    row_str = ""
    for j in range(d):
        # row_str += f" {v_grid[-1][j]}" # N
        row_str += f" {xframe[d*d+d*(d-1)+j]:d}" # N
    print(row_str)

    for i in range(d):

        row_str = ""
        for j in range(d):
            # row_str += f"{h_grid[i][j]}{syn_grid[i][j]}"
            # row_str += f"{h_grid[i][j]}{syndrome[d*i+j]}"
            row_str += f"{xframe[d*i+j]:d}{syndrome[d*i+j]:d}"
        # row_str += f"{h_grid[i][0]}"
        row_str += f"{xframe[d*i+0]:d}"
        print(row_str)

        row_str = ""
        for j in range(d):
            # row_str += f" {v_grid[i][j]}" # N
            row_str += f" {xframe[d*d+d*i+j]:d}" # N
        print(row_str)
    print()
        
def print_evolution(d, xframe, n_steps):

    H = toric_code_z_stabilisers(d)
    CA = harrington.CA(d)

    syndrome = H @ xframe % 2
    print_toric_code(d,xframe,syndrome)
    for i in range(n_steps):
        syndrome = H @ xframe % 2
        CA.step(xframe, syndrome)
        syndrome = H @ xframe % 2
        print_toric_code(d,xframe,syndrome)

# def simple_log_test(d, pauli_frame_x): # identical to check logicals via dot product
#     # Check logical-X error on the code.
#     # Check top and right boundaries for odd crossings
#     v_grid = pauli_frame_x[:d**2].reshape((d,d))
#     h_grid = pauli_frame_x[d**2:].reshape((d,d))
#     return np.sum(v_grid[:,0]) % 2 or np.sum(h_grid[0,:]) % 2

def gen_err_conf(n,p):
    return (np.random.random(n) < p).astype(bool)

def gen_err_conf_w(n,w):
    conf = np.zeros(n, dtype=bool)
    ids = np.random.choice(n, int(w), replace=False)
    conf[ids] = 1
    return conf

def gen_err_conf_lt_w(n,w):
    w_ = np.random.choice(int(w))
    conf = np.zeros(n, dtype=bool)
    ids = np.random.choice(n, w_, replace=False)
    conf[ids] = 1
    return conf

def benchmark_code_capacity(d, p, N):
    # path = Path(f"./data/cc/d={d}_p={p:.4f}.csv")
    # path = Path(f"./data/cc2/d={d}_p={p:.4f}.csv")
    path = Path(f"./data/cc_U=16/d={d}_p={p:.4f}.csv")
    # path = Path(f"./data/cc_U=10/d={d}_p={p:.4f}.csv")
    # path = Path(f"./data/cc_harRules/d={d}_p={p:.4f}.csv")
    print(path)
    path.touch(exist_ok=True)

    H = toric_code_z_stabilisers(d)
    logicals = toric_code_z_logicals(d)

    max_correctable_errs = 2 ** (np.log2(d) / np.log2(3))
    # print(f"Max. correctable errors: {max_correctable_errs}")

    n_log_errs = 0
    for i in range(N):

        CA = harrington.CA(d)
        xframe = gen_err_conf(2*d*d,p)
        # xframe = gen_err_conf_w(2*d*d,max_correctable_errs - 1)
        # xframe = gen_err_conf_lt_w(2*d*d,max_correctable_errs - 1)

        # for checking error confs
        n_errs = np.sum(xframe)
        xframe_ = np.copy(xframe.astype(int))

        syndrome = H @ xframe % 2

        log_err = 0
        while np.sum(syndrome) != 0.0:
            syndrome = H @ xframe % 2
            CA.step(xframe, syndrome)
        logicals_flipped = logicals @ xframe % 2
        log_err = int( np.sum(logicals_flipped) > 0.0 )

        # check error weight
        if log_err and n_errs < max_correctable_errs:
            print(f"Fail for conf of {n_errs} errors for d={d}")
            print(xframe_.tolist())
            # hgrid, vgrid = xframe_[:d*d], xframe_[d*d:]
            # print('hgrid:')
            # print(hgrid.reshape((d,d)))
            # print('vgrid:')
            # print(vgrid.reshape((d,d)))

            
        n_log_errs += log_err

        with open(path, 'a', newline='\n') as fd:
            writer = csv.writer(fd)
            writer.writerow([log_err])

    return n_log_errs

def benchmark_pheno(d, p, N): 

    path = Path(f"./data/pheno_U=16/d={d}_p={p:.4f}.csv")
    # path = Path(f"./data/pheno/d={d}_p={p:.4f}.csv")
    # path = Path(f"./data/pheno_no_cor/d={d}_p={p:.4f}.csv")
    print(path)
    path.touch(exist_ok=True)

    H = toric_code_z_stabilisers(d)
    logicals = toric_code_z_logicals(d)
    matching = Matching.from_check_matrix(H, weights=np.log((1-p)/p), faults_matrix=logicals)

    T = 0 
    for i in range(N):

        CA = harrington.CA(d)
        xframe = np.zeros((2*d*d), dtype=bool)
        t = 0 
        while True:
            t += 1

            xframe = xframe ^ gen_err_conf(2*d*d,p) # errors add up.
            syndrome = H @ xframe % 2 

            predicted_logicals_flipped = matching.decode(syndrome)
            actual_logicals_flipped = logicals @ xframe % 2 
            if not np.array_equal(predicted_logicals_flipped, actual_logicals_flipped):
                break

            CA.step(xframe, syndrome)

        with open(path, 'a', newline='\n') as fd: 
            writer = csv.writer(fd)
            writer.writerow([t])

        T += t
    return T

def benchmark_pheno_q(d, p, N):

    path = Path(f"./data/pheno_U=16_q/d={d}_p={p:.4f}.csv")
    # path = Path(f"./data/pheno_q/d={d}_p={p:.4f}.csv")
    print(path)
    path.touch(exist_ok=True)

    H = toric_code_z_stabilisers(d)
    logicals = toric_code_z_logicals(d)
    matching = Matching.from_check_matrix(H, weights=np.log((1-p)/p), faults_matrix=logicals)

    T = 0
    for i in range(N):

        CA = harrington.CA(d)
        t = 0
        xframe = np.zeros((2*d*d), dtype=bool)
        while True:
            t += 1

            syndrome = H @ xframe % 2

            predicted_logicals_flipped = matching.decode(syndrome)
            actual_logicals_flipped = logicals @ xframe % 2
            if not np.array_equal(predicted_logicals_flipped, actual_logicals_flipped):
                break

            syndrome = syndrome ^ gen_err_conf(d*d,p) # meas. noise
            CA.step(xframe, syndrome)

        with open(path, 'a', newline='\n') as fd:
            writer = csv.writer(fd)
            writer.writerow([t])

        T += t
    return T

def benchmark_pheno_q_psig(d, p, N): # p=q=psig

    path = Path(f"./data/pheno_U=16_q_psig/d={d}_p={p:.4f}.csv")
    # path = Path(f"./data/pheno_q/d={d}_p={p:.4f}.csv")
    print(path)
    path.touch(exist_ok=True)

    H = toric_code_z_stabilisers(d)
    logicals = toric_code_z_logicals(d)
    matching = Matching.from_check_matrix(H, weights=np.log((1-p)/p), faults_matrix=logicals)

    T = 0
    for i in range(N):

        CA = harrington.CA(d)
        t = 0
        xframe = np.zeros((2*d*d), dtype=bool)
        while True:
            t += 1

            xframe = xframe ^ gen_err_conf(2*d*d,p) # data noise
            syndrome = H @ xframe % 2

            predicted_logicals_flipped = matching.decode(syndrome)
            actual_logicals_flipped = logicals @ xframe % 2
            if not np.array_equal(predicted_logicals_flipped, actual_logicals_flipped):
                break

            syndrome = syndrome ^ gen_err_conf(d*d,p) # meas. noise
            CA.step(xframe, syndrome, p) # signal noise

        with open(path, 'a', newline='\n') as fd:
            writer = csv.writer(fd)
            writer.writerow([t])

        T += t
    return T

def benchmark_pheno_q_pcs(d, p, N): # p=q=psig

    path = Path(f"./data/pheno_U=16_q_pcs/d={d}_p={p:.4f}.csv")
    # path = Path(f"./data/pheno_q/d={d}_p={p:.4f}.csv")
    print(path)
    path.touch(exist_ok=True)

    H = toric_code_z_stabilisers(d)
    logicals = toric_code_z_logicals(d)
    matching = Matching.from_check_matrix(H, weights=np.log((1-p)/p), faults_matrix=logicals)

    T = 0
    for i in range(N):

        CA = harrington.CA(d)
        t = 0
        xframe = np.zeros((2*d*d), dtype=bool)
        while True:
            t += 1

            xframe = xframe ^ gen_err_conf(2*d*d,p) # data noise
            syndrome = H @ xframe % 2

            predicted_logicals_flipped = matching.decode(syndrome)
            actual_logicals_flipped = logicals @ xframe % 2
            if not np.array_equal(predicted_logicals_flipped, actual_logicals_flipped):
                break

            syndrome = syndrome ^ gen_err_conf(d*d,p) # meas. noise
            CA.step(xframe, syndrome, p) # signal noise

        with open(path, 'a', newline='\n') as fd:
            writer = csv.writer(fd)
            writer.writerow([t])

        T += t
    return T

# ds = [3,9,27,81]
# ds = [27,81]
# ds = [3,9,27]
# ds = [27]
ds = [81]
# ds = [9]
# ds = [3]
# ds = [3,9,27,81]
# ds = [81]
# ps1 = np.linspace(1, 1e-1,10)[1:]
# ps2 = np.linspace(1e-1, 1e-2,10)[1:]
# ps3 = np.linspace(1e-2, 1e-3,10)[1:]
# ps = np.concatenate([ps1, ps2, ps3])
# ps = np.concatenate([ ps2, ps3])
# ps = np.linspace(1e-4, 1e-3,10)[::-1][1:]
# ps = [4e-3]
ps = [2e-2]
# ps = [1.5e-3]
# ps = [1e-3,2e-3,3e-3,4e-3,5e-3]
# ps = [1e-3,2e-3,3e-3,4e-3,5e-3]
# ps = [1e-1, 9e-2, 8e-2, 7e-2]
# ps = [5e-1, 4e-1, 3e-1, 2e-1, 1e-1]
# ps = [4e-2,5e-2,6e-2,7e-2,8e-2,9e-2,1e-1]
# ps = [4e-3, 3e-3,2e-3,1e-3]
# ps = [1e-1,9e-2,8e-2,7e-2,6e-2,5e-2]
# ps = [6e-2,7e-2,8e-2,9e-2,1e-1]
# ps = [1e-2, 9e-3, 8e-3,7e-3]
# ps = [2e-3, 1e-3]
# ps = [4e-2, 3e-2, 2e-2]
# ps = [1e-2,9e-3,8e-3]
# ps = [4.5e-2]
# ps = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3]
# ps = [1e-2, 2e-2, 3e-2]
# ps = [1e-1,9e-2,8e-2,7e-2,6e-2,5e-2]
# ps = [9e-4,8e-4,7e-4,6e-4,5e-4,4e-4,3e-4,2e-4,1e-4]
# ps = [9e-3,8e-3,7e-3,6e-3,5e-3]
N = 1_000_000
# N = 100

for d in ds:
    print(f"--- d={d} ---")
    for p in ps:
        n = benchmark_code_capacity(d, p, N)
        # n = benchmark_pheno(d, p, N)
        # n = benchmark_pheno_q(d, p, N)
        # n = benchmark_pheno_q_psig(d, p, N)
        # n = benchmark_pheno_q_pcs(d, p, N)
        print(f"- p={p:.4f}, mu={n/N:}")


# xframe = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
# xframe = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
# xframe = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
# xframe = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# d = 9
# xframe = np.zeros((2*d*d), dtype=bool)
# xframe[4] = 1

# print_evolution(9,xframe,30)
