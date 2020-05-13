from mpi4py import MPI
import numpy as np
import time

def df(x, A, b):
    return A.T @ (A @ x - b)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

M, batch = 2, 50
A, b = np.zeros([batch, M]), np.zeros([batch])
AA, bb = None, None

if rank == 0:
    # np.random.seed(0)
    # Generate data on node 0
    AA = np.random.randn(size*batch, M) * 0.01
    bb = np.random.randn(size*batch)

    print("Optimal value:", np.linalg.solve(AA.T @ AA, AA.T @ bb), flush=True)
    
    AA = AA.reshape([size, batch, M])
    bb = bb.reshape([size, batch])


comm.Scatter(AA, A, root=0)
comm.Scatter(bb, b, root=0)


mem = np.zeros([M], dtype=np.float)
x = np.zeros([M], dtype=np.float)
win = MPI.Win.Create(mem, comm=comm)

MAX_ITER = 5000 + 1
STOP_FLAG, i, lr = False, 0, 1

# if rank == 0:
#     log = np.zeros([MAX_ITER, M])

while True:
    if STOP_FLAG or i >= MAX_ITER:
        break
    win.Lock(0)
    win.Get(x, 0)
    # if rank == 0:
    #     log[i] = x
    win.Accumulate(-lr/size*df(x, A, b), target_rank=0, op=MPI.SUM)
    win.Unlock(0)
    i += 1

if rank == 0:
    print(x, flush=True)
    # np.save("tmp/async.npy", log)

win.Free()