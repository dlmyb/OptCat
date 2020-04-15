from mpi4py import MPI
import numpy as np
import random
from collections import Counter
from helper import scatter

r = random.Random()

def df(x, A, b):
    return A.T @ (A @ x - b)


def eval_theta(L, m, lamba, eta, tau):
    t1 = 4*L * (eta + L*tau*tau*eta*eta) / (1-2*L*L*eta*eta*tau*tau)
    print(t1)
    return (1/lamba/eta/m + t1) / (1 - t1)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
        
M, batch = 5, 5
L, mu = 0, 0
AA, bb = None, None

dsts = np.hstack([0, np.ones(size-1, np.int)])
# dsts = np.arange(size)

if rank == 0:
    AA = np.random.randn(np.sum(dsts)*batch, M) * 0.01
    bb = np.random.randn(np.sum(dsts)*batch, 1) * 0.01
    eigs = np.linalg.eigvals(AA.T @ AA)
    L, mu = np.max(eigs), np.min(eigs)

    print("Optimal value:", np.linalg.solve(AA.T @ AA, AA.T @ bb).flatten())
    print("L, mu:", L, mu)

A = scatter(AA, dsts)
b = scatter(bb, dsts).flatten()

x = np.zeros(M)
if rank == 0:
    x = np.random.randn(M) * 5
win = MPI.Win.Create(x, comm=comm)

MAX_ITER = 3000 + 1 # +1 could print info when loop ends.
EPOCH = 16
TAU = 4

if rank == 0:
    lr = 100
    print("theta:", eval_theta(L, EPOCH, mu, lr, TAU))
    buf = np.zeros([size-1, M])
    df_new = np.zeros(M)
    x_stored = np.copy(x)
    
    dsts = np.arange(1, size)
    # dists = Counter()
    for i in range(MAX_ITER):
        x -= x - x_stored # x = x_stored
        for j in range(size-1):
            comm.send(1, j+1)
            comm.Recv(buf[j], j+1)

        sample = r.randint(a=0, b=EPOCH-1)
        j = 0
        while j < EPOCH:
            dst_round = r.choices(dsts, k=min(TAU, EPOCH-j))

            # Send a wake up signal
            reqs = []
            for dst in dst_round:
                reqs.append(comm.isend(1, dst))
            MPI.Request.Waitall(reqs)

            for k in range(min(TAU, EPOCH-j)):
                st = MPI.Status()
                comm.probe(status=st)
                dst = st.Get_source()
                df_last = buf[dst-1]
                comm.Recv(df_new, dst)
                x -= lr*(df_new - df_last + np.sum(buf, axis=0)/size)

                if sample == j:
                    x_stored -= x_stored - x
                j += 1

        if i % 100 == 0:
            print(i, x)
    
    for dst in dsts:
        # Tell all workers stop
        comm.send(0, dst)

    # print(dists)

else:
    x = np.zeros(M)
    while comm.recv(source=0):
        win.Lock(0, lock_type=MPI.LOCK_SHARED)
        win.Get(x, 0)
        win.Unlock(0)
        comm.Send(df(x, A, b), 0)

