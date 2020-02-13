from mpi4py import MPI
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import time

def ring_matrix(n):
    w = np.eye(n) * 0.5
    for i in range(n):
        w[i, (i+1)%n] = 0.25
        w[i, (i-1)%n] = 0.25
    return w

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class GCon(object):
    """
    Decentralized consensus module based on MPI
    """
    def __init__(self, features, w):
        self.x_last = np.zeros(features)
        self.s_last = np.zeros(features)
        self.w = w
        # target and buf are for exchanging data
        self.target = []
        self.rank = MPI.COMM_WORLD.Get_rank()
        for i, ws in enumerate(w):
            if ws != 0 and i != self.rank:
                self.target.extend([i])
        self.buf = np.zeros([len(self.target), features])
    
    def __call__(self, x):
        requests = []
        for i, tar in enumerate(self.target):
            requests.extend([
                comm.Isend(self.s_last * self.w[tar], tar), 
                comm.Irecv(self.buf[i], tar)
            ])
        # Wait all requests finished
        MPI.Request().Waitall(requests)
        # Updating formula, see Shuo Han's CSL letter, formula (3)
        result = np.sum(self.buf, axis=0) + self.w[self.rank] * \
            self.s_last + (x - self.x_last)
        self.x_last, self.s_last = x, result
        return result



M, batch = 2, 20
A = np.zeros([batch*2, M+1])
AA, WW = None, None
W = np.zeros(size)

if rank == 0:
    # Generate data points
    np.random.seed(0)
    N = batch * size
    X1 = np.random.normal(scale=[1, 0.8], size=[N, M]) + np.array([0, 1.5])
    X2 = np.random.normal(scale=[1, 0.5], size=[N, M]) + np.array([1.5, 0])
    # np.save("tmp/data.npy", np.stack([X1, X2]))

    AA = np.vstack([
        -1 * np.hstack([X1, np.ones([N, 1])]),
        np.hstack([X2, np.ones([N, 1])])
    ])

    random_indexes = np.random.permutation(N).reshape([size, -1])
    random_indexes = np.hstack([random_indexes, random_indexes + N])

    AA = np.stack([AA[random_indexes[idx], :] for idx in range(size)])

    # Generate W, here we use ring.
    WW = ring_matrix(size)

comm.Scatter(AA, A, root=0)
comm.Scatter(WW, W, root=0)


# Training period

MAX_ITER = 1000 + 1
ABSTOL = 1e-4
RELTOL = 1e-2
rho, C = 1, 1

z, u, x = np.zeros(M+1), np.zeros(M+1), np.zeros(M+1)
x_con = GCon(M+1, W)
u_con = GCon(M+1, W)

STOP_FLAG, i = False, 0

while True:
    if STOP_FLAG or i >= MAX_ITER:
        break
    
    x_var = cp.Variable(M+1)
    cp.Problem(
        cp.Minimize(cp.sum(cp.pos(cp.matmul(A, x_var)+1)) + \
        rho/2 * cp.sum_squares(x_var-z+u))
    ).solve(verbose=False, solver=cp.MOSEK)
    x = x_var.value

    x_ave = x_con(x)
    z = size * rho/(1/C + size*rho) * (x_ave + u_con(u))
    u += x - z
    
    if i % 50 == 0:
        print(rank, i, x_ave, u, z)

    i += 1