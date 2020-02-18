from mpi4py import MPI
import numpy as np
from consensus import GCon, matrix
from scipy.linalg import lu_factor, lu_solve

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

M, batch = 2, 50
A, b = np.zeros([batch, M]), np.zeros([batch])
AA, WW, bb = None, None, None
W = np.zeros(size)

if rank == 0:
    np.random.seed(0)
    # Generate data on node 0
    AA = np.random.randn(size*batch, M)
    bb = np.random.randn(size*batch)

    print("Optimal value:", np.linalg.inv(AA.T.dot(AA)).dot(AA.T).dot(bb))
    
    AA = AA.reshape([size, batch, M])
    bb = bb.reshape([size, batch])

    # Generate W, here we use ring.
    WW = matrix.ring_matrix(size)

comm.Scatter(AA, A, root=0)
comm.Scatter(bb, b, root=0)
comm.Scatter(WW, W, root=0)

MAX_ITER = 500 + 1
rho = 1

# x = np.zeros([M])
y = np.zeros([M])
x_con = GCon(M, W)
x_ave = np.zeros([M])
STOP_FLAG, i = False, 0

lu, piv = lu_factor((A.T @ A) + rho * np.eye(M))
ATb = A.T @ b

while True:
    if STOP_FLAG or i >= MAX_ITER:
        break

    x = lu_solve((lu, piv), ATb + x_ave - y)
    x_ave = x_con(x)
    y = y + rho * (x - x_ave)

    if i % 50 == 0:
        print(i, rank, x_ave, y)
    
    i += 1

