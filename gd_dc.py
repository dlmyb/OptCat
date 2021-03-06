from consensus import GCon, matrix
from mpi4py import MPI
import numpy as np
from consensus import GCon, matrix

def df(x, A, b):
    return A.T.dot(A.dot(x) - b)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

M, batch = 2, 50
A, b = np.zeros([batch, M]), np.zeros([batch])
W = np.zeros(size)
AA, WW, bb = None, None, None

if rank == 0:
    np.random.seed(0)
    AA = np.random.randn(size*batch, M)
    bb = np.random.randn(size*batch)

    print("Optimal value:", np.linalg.solve(AA.T @ AA, AA.T @ bb))
    
    AA = AA.reshape([size, batch, M])
    bb = bb.reshape([size, batch])
    WW = matrix.ring_matrix(size)

comm.Scatter(AA, A, root=0)
comm.Scatter(bb, b, root=0)
comm.Scatter(WW, W, root=0)

MAX_ITER = 600 + 1
STOP_FLAG, i, lr = False, 0, 0.001

x = np.zeros([M])
# dx = np.zeros([M])
x_con = GCon(M, W)
dx_con = GCon(M, W)
x_hat = np.zeros([M])
# dx_hat = np.zeros([M])

while True:
    if STOP_FLAG or i >= MAX_ITER:
        break
    dx = df(x_hat, A, b)
    dx_hat = dx_con(dx)

    x -= lr * dx_hat
    x_hat = x_con(x)

    if i % 100 == 0:
        print(i, rank, x_hat, dx_hat)

    i += 1
