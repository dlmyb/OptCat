from mpi4py import MPI
import numpy as np
import random
from collections import Counter
from helper import scatter

r = random.Random()

def df(x, A, b):
    return A.T @ (A @ x - b)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


M, batch = 5, 5
L, mu = 0, 0
AA, bb = None, None

dsts = np.hstack([0, np.ones(size-1, np.int)])

if rank == 0:
    AA = np.random.randn(np.sum(dsts)*batch, M) * 5
    bb = np.random.randn(np.sum(dsts)*batch, 1) * 5
    
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

MAX_ITER = 200 + 1 # +1 could print info when loop ends.
EPOCH = 32

if rank == 0:
    lr = 0.1/L
    buf = np.zeros([size-1, M])
    df_new = np.zeros(M)
    
    dst_range = np.arange(1, size)
    # dists = Counter()
    for i in range(MAX_ITER):
        for j in range(size-1):
            comm.send(1, j+1)
            comm.Recv(buf[j], j+1)
        for j in range(EPOCH):
            dst = r.choice(dst_range)
            # dists.update([dst])
            # Send a wake up signal
            comm.send(1, dst)
            df_last = buf[dst-1]
            comm.Recv(df_new, dst)

            x -= lr*(df_new - df_last + np.sum(buf, axis=0)/size)

        if i % 10 == 0:
            print(i, x)
    
    for dst in dst_range:
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

