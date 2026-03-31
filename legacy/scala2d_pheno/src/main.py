import scala
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
    CA = scala.CA(d)

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

def benchmark_pheno_var_reset(d, p, N):

    # path = Path(f"./data/pheno/d={d}_p={p:.4f}.npy")
    # path = Path(f"./data/pheno3/d={d}_p={p:.4f}.npy")

    t_r_max = int((d+1)/2)
    # t_r_max = 23
    # t_r_max = 10

    # for t_r in range(1, t_r_max+1):
    # for t_r in range(8, t_r_max+1):
    for t_r in [10]:

        path = Path(f"./data/pheno_var_reset/d={d}_p={p:.4f}_t={t_r}.npy")
        print(path)
        # path.touch(exist_ok=True)
        if path.exists():
            arr = np.load(path)
        else:
            arr = np.array([], dtype=int)

        H = toric_code_z_stabilisers(d)
        logicals = toric_code_z_logicals(d)
        matching = Matching.from_check_matrix(H, weights=np.log((1-p)/p), faults_matrix=logicals)

        T = 0
        T_prev = 0
        for i in range(N):

            CA = scala.CA(d)
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

                if t % t_r == 0:
                    CA.reset()

            T += t

            arr = np.concatenate([arr, [t]])
            np.save(path, arr)

        print(f"T_R={t_r} -- {T/N}")
        if T_prev > T: # exit once we reached the peak.
            return T
        T_prev = T

    return T

def benchmark_pheno_q_var_reset(d, p, N):

    # path = Path(f"./data/pheno/d={d}_p={p:.4f}.npy")
    # path = Path(f"./data/pheno3/d={d}_p={p:.4f}.npy")

    # HERERERER

    # t_r_max = int((d+3)/2)
    # t_r_max = int(d)
    # t_r_max = 23
    t_r_min = 6
    t_r_max = 7

    for t_r in range(t_r_min, t_r_max+1):
    # # for t_r in range(8, t_r_max+1):
    # for t_r in [5]:

        path = Path(f"./data/pheno_q_var_reset/d={d}_p={p:.4f}_t={t_r}.npy")
        print(path)
        # path.touch(exist_ok=True)
        if path.exists():
            arr = np.load(path)
        else:
            arr = np.array([], dtype=int)

        H = toric_code_z_stabilisers(d)
        logicals = toric_code_z_logicals(d)
        matching = Matching.from_check_matrix(H, weights=np.log((1-p)/p), faults_matrix=logicals)

        T = 0
        T_prev = 0
        for i in range(N):

            CA = scala.CA(d)
            xframe = np.zeros((2*d*d), dtype=bool)

            t = 0
            while True:
                t += 1

                # xframe = xframe ^ gen_err_conf(2*d*d,p) # errors add up.
                syndrome = H @ xframe % 2

                predicted_logicals_flipped = matching.decode(syndrome)
                actual_logicals_flipped = logicals @ xframe % 2
                if not np.array_equal(predicted_logicals_flipped, actual_logicals_flipped):
                    break

                syndrome = syndrome ^ gen_err_conf(d*d,p) # meas. noise
                CA.step(xframe, syndrome)

                if t % t_r == 0:
                    CA.reset()

            T += t

            arr = np.concatenate([arr, [t]])
            np.save(path, arr)

        print(f"T_R={t_r} -- {T/N}")
        if T_prev > T: # exit once we reached the peak.
            return T
        T_prev = T

    return T

def benchmark_pheno_psig_var_reset(d, p, psig, N):

    t_r_max = int( (d+1)/2 )

    # for t_r in range(1, t_r_max+1):
    for t_r in [7,8]:

        path = Path(f"./data/pheno_psig_var_reset/d={d}_p={p:.4f}_psig={psig:.4f}_t={t_r}.csv")
        print(path)
        path.touch(exist_ok=True)

        H = toric_code_z_stabilisers(d)
        logicals = toric_code_z_logicals(d)
        matching = Matching.from_check_matrix(H, weights=np.log((1-p)/p), faults_matrix=logicals)

        T = 0
        for i in range(N):

            CA = scala.CA(d)
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

                # syndrome = syndrome ^ gen_err_conf(d*d,p) # meas. noise
                CA.step(xframe, syndrome, psig)

                if t % t_r == 0:
                    CA.reset()

            with open(path, 'a', newline='\n') as fd:
                writer = csv.writer(fd)
                writer.writerow([t])

            T += t

def benchmark_pheno_q_psig_var_reset(d, p, psig, N):

    # t_r_max = int( (d+1)/2 )
    t_r_min = 25 
    t_r_max = 25

    for t_r in range(t_r_min, t_r_max+1):

        path = Path(f"./data/pheno_q_psig_var_reset/d={d}_p={p:.4f}_psig={psig:.4f}_t={t_r}.csv")
        print(path)
        path.touch(exist_ok=True)

        H = toric_code_z_stabilisers(d)
        logicals = toric_code_z_logicals(d)
        matching = Matching.from_check_matrix(H, weights=np.log((1-p)/p), faults_matrix=logicals)

        T = 0
        for i in range(N):

            CA = scala.CA(d)
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
                CA.step(xframe, syndrome, p) # sig noise

                if t % t_r == 0:
                    CA.reset()

            with open(path, 'a', newline='\n') as fd:
                writer = csv.writer(fd)
                writer.writerow([t])

            T += t

def benchmark_pheno(d, p, N):

    # path = Path(f"./data/pheno/d={d}_p={p:.4f}.npy")
    # path = Path(f"./data/pheno3/d={d}_p={p:.4f}.npy")
    path = Path(f"./data/pheno_n/d={d}_p={p:.4f}.npy")
    print(path)
    # path.touch(exist_ok=True)
    if path.exists():
        arr = np.load(path)
    else:
        arr = np.array([], dtype=int)

    H = toric_code_z_stabilisers(d)
    logicals = toric_code_z_logicals(d)
    matching = Matching.from_check_matrix(H, weights=np.log((1-p)/p), faults_matrix=logicals)

    T = 0
    for i in range(N):

        CA = scala.CA(d)
        xframe = np.zeros((2*d*d), dtype=bool)

        # t_r = 2;
        # reset_interval = 3;

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

            # if t % int((d+1)/2) == 0:
            # if t % 2 == 0:
            # if t % 5 == 0:
            # if t % 10 == 0:
            if t % 23 == 0:
                CA.reset()
            # if t == t_r:
            #     CA.reset()
            #     # reset_interval = (reset_interval + 1) % d # there is a problem here!
            #     reset_interval += 1
            #     if reset_interval == d: 
            #     # if reset_interval == int((d+1)/2):
            #         reset_interval = 3
            #     # reset_interval = (reset_interval + 1) % int((d+1)/2)
            #     t_r = t + reset_interval

        T += t

        arr = np.concatenate([arr, [t]])
        np.save(path, arr)

    return T

def benchmark_pheno_q(d, p, N):

    path = Path(f"./data/pheno_q/d={d}_p={p:.4f}.npy")
    print(path)

    if path.exists():
        arr = np.load(path)
    else:
        arr = np.array([], dtype=int)

    H = toric_code_z_stabilisers(d)
    logicals = toric_code_z_logicals(d)
    matching = Matching.from_check_matrix(H, weights=np.log((1-p)/p), faults_matrix=logicals)

    T = 0
    for i in range(N):

        CA = scala.CA(d)
        xframe = np.zeros((2*d*d), dtype=bool)

        t_r = 2;
        reset_interval = 3;
        t_r_max = d*d + d - 2;

        t = 0
        while True:
            t += 1

            # xframe = xframe ^ gen_err_conf(2*d*d,p) # errors add up.
            syndrome = H @ xframe % 2

            predicted_logicals_flipped = matching.decode(syndrome)
            actual_logicals_flipped = logicals @ xframe % 2
            if not np.array_equal(predicted_logicals_flipped, actual_logicals_flipped):
                break

            syndrome = syndrome ^ gen_err_conf(d*d,p) # meas. noise
            CA.step(xframe, syndrome)

            # if t % int((d+1)/2) == 0:
            #     CA.reset()
            if t == t_r:
                CA.reset()
                # reset_interval = (reset_interval + 1) % d # there is a problem here!
                reset_interval += 1
                # if reset_interval == int((d-1)/2):
                if reset_interval == int((d+1)/2):
                    reset_interval = 3
                # reset_interval = (reset_interval + 1) % int((d+1)/2)
                t_r = t + reset_interval

        T += t

        arr = np.concatenate([arr, [t]])
        np.save(path, arr)

    return T


# def benchmark_pheno_psig_pq(d, p, psig, N):

#     path = Path(f"./data/pheno_psig_pq/d={d}_p={p:.4f}_psig={psig:.4f}.csv")
#     print(path)
#     path.touch(exist_ok=True)

#     H = toric_code_z_stabilisers(d)
#     logicals = toric_code_z_logicals(d)
#     matching = Matching.from_check_matrix(H, weights=np.log((1-p)/p), faults_matrix=logicals)

#     T = 0
#     for i in range(N):

#         CA = scala.CA(d)
#         t = 0
#         xframe = np.zeros((2*d*d), dtype=bool)

#         while True:
#             t += 1

#             xframe = xframe ^ gen_err_conf(2*d*d,p) # data noise

#             syndrome = H @ xframe % 2

#             predicted_logicals_flipped = matching.decode(syndrome)
#             actual_logicals_flipped = logicals @ xframe % 2
#             if not np.array_equal(predicted_logicals_flipped, actual_logicals_flipped):
#                 break

#             syndrome = syndrome ^ gen_err_conf(d*d,p) # meas. noise
#             CA.step(xframe, syndrome, psig)

#         with open(path, 'a', newline='\n') as fd:
#             writer = csv.writer(fd)
#             writer.writerow([t])

#         T += t
#     return T

def save_orbit(d, p, N):

    path = Path(f"./data/orbit/d={d}_p={p:.4f}.csv")
    print(path)
    path.touch(exist_ok=True)

    H = toric_code_z_stabilisers(d)
    logicals = toric_code_z_logicals(d)
    matching = Matching.from_check_matrix(H, weights=np.log((1-p)/p), faults_matrix=logicals)

    xframe = np.zeros((2*d*d), dtype=bool)
    # xframe = xframe ^ gen_err_conf(2*d*d,p) # data noise
    CA = scala.CA(d)

    for i in range(N):
        xframe = xframe ^ gen_err_conf(2*d*d,p) # data noise
        syndrome = H @ xframe % 2
        CA.step(xframe, syndrome)
        syndrome = H @ xframe % 2

        with open(path, 'a', newline='\n') as fd:
            writer = csv.writer(fd)
            writer.writerow(["".join(map(str, syndrome))])

            
# ds = [3,5,7,9,11,13,15]
ds = [15]
# ps = np.concatenate([np.linspace(1, 1e-1,10)[1:-1], np.linspace(1e-2, 1e-1,10)[::-1], np.linspace(1e-2, 1e-3,10)[1:]])
# ps = np.concatenate([np.linspace(1, 1e-1,10)[1:-1], np.linspace(1e-2, 1e-1,10)[::-1]])
# ps = np.linspace(1, 1e-1,10)[1:-1]
# ps = np.linspace(1e-2, 1e-1,10)[::-1]
# ps = np.linspace(1e-2, 1e-3,10)[1:]
# ps = np.linspace(1e-3, 1e-4,10)[2:]
# ps = [1e-1, 9e-2, 8e-2, 7e-2]
# ps = [2e-2,1e-2]
# ps = [9e-3]
# ps = [6e-3,5e-3,4e-3,3e-3,2e-3,1e-3]
# ps = [8e-3,7e-3,6e-3,5e-3,4e-3,3e-3,2e-3,1e-3]
ps = [6e-3,5e-3,4e-3,3e-3,2e-3,1e-3]
# ps = [9e-1, 8e-1, 7e-1, 6e-1, 5e-1,4e-1,3e-1,2e-1,1e-1]#, 9e-2, 8e-2, 7e-2, 6e-2, 5e-2, 4e-2, 3e-2, 2e-2, 1e-2,]
# ps = [9e-2, 8e-2, 7e-2, 6e-2, 5e-2,4e-2,3e-2,2e-2,1e-2]#, 9e-2, 8e-2, 7e-2, 6e-2, 5e-2, 4e-2, 3e-2, 2e-2, 1e-2,]

# ps = [
#     9e-1, 8e-1, 7e-1, 6e-1, 5e-1,4e-1,3e-1,2e-1,1e-1,
#     9e-2, 8e-2, 7e-2, 6e-2, 5e-2, 4e-2, 3e-2, 2e-2, 1e-2,
#     9e-3, 8e-3, 7e-3, 6e-3, 5e-3, 4e-3, 3e-3, 2e-3, 1e-3,
#      ]

N = 100

for d in ds:
    print(f"--- d={d} ---")
    for p in ps:
        # save_orbit(d,p,N)
        # n = benchmark_pheno(d, p, N)
        # n = benchmark_pheno_var_reset(d, p, N)
        n = benchmark_pheno_q_var_reset(d, p, N)
        # n = benchmark_pheno_psig_var_reset(d, p, p, N)
        # n = benchmark_pheno_q_psig_var_reset(d, p, p, N)
        # n = benchmark_pheno_q(d, p, N)
        # print(f"- p={p:.4f}, mu={n/N:}")
