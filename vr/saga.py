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
AA, bb = None, None

# dsts = np.arange(size, dtype=np.int)
dsts = np.hstack([0, np.ones(size-1, np.int)])

if rank == 0:
    # np.random.seed(0)
    # AA = np.random.randn(600, M) * 5
    # bb = np.random.randn(600, 1) * 5
    AA = np.random.randn(np.sum(dsts)*batch, M) * 5
    bb = np.random.randn(np.sum(dsts)*batch, 1) * 5

    print("Optimal value:", np.linalg.solve(AA.T @ AA, AA.T @ bb).flatten())

A = scatter(AA, dsts)
b = scatter(bb, dsts).flatten()

x = np.zeros([M])
if rank == 0:
    x = np.random.randn(M) * 5
comm.Bcast(x, root=0)
win = MPI.Win.Create(x, comm=comm)

MAX_ITER = 200 + 1
lr = 0.05 / size

if rank == 0:
    buf = np.zeros([size-1, M])
    df_new = np.zeros([M])

    reqs = []
    for i in range(size-1):
        reqs.append(comm.Irecv(buf[i], source=i+1))
    MPI.Request.Waitall(reqs)
    
    dst_range = np.arange(1, size) # Uniformly sample.
    dists = Counter()
    for i in range(size*MAX_ITER):

        dst = r.choice(dst_range)
        dists.update([dst])

        # Send a wake up signal
        comm.send(1, dst)

        df_last = buf[dst-1]
        comm.Recv(df_new, dst)
        buf[dst-1] = df_new

        x -= lr*(df_new - df_last + np.sum(buf, axis=0)/size)

        if i % 200 == 0:
            print(i, x)
    
    for dst in np.unique(dst_range):
        # Tell all workers stop
        comm.send(0, dst)
    print(dists)


else:
    x = np.zeros([M]) + 5
    comm.Send(df(x, A, b), 0)
    while comm.recv(source=0):
        win.Lock(0)
        win.Get(x, 0)
        win.Unlock(0)
        comm.Send(df(x, A, b), 0)

