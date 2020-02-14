from mpi4py import MPI
import cvxpy as cp
import numpy as np
from consensus import mpireduce

norm, sqrt = np.linalg.norm, np.sqrt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def hinge_loss(A, x):
    p = 1 + A.dot(x)
    p[p < 0] = 0
    return np.sum(1 + p)

def objective(A, C, x, z):
    return hinge_loss(A, x) + 1/(2*C)*np.sum(np.square(z))

def max_sigma(X, **kwargs):
    _u, s, _vh = np.linalg.svd(X, full_matrices=False)
    return s[0]


M, batch = 2, 10
A = np.zeros([batch*2, M+1])
AA = None

if rank == 0:
    np.random.seed(0)
    N = batch * size
    X1 = np.random.normal(scale=[1, 0.8], size=[N, M]) + np.array([0, 1.5])
    X2 = np.random.normal(scale=[1, 0.5], size=[N, M]) + np.array([1.5, 0])
    np.save("tmp/data.npy", np.stack([X1, X2]))

    AA = np.vstack([
        -1 * np.hstack([X1, np.ones([N, 1])]),
        np.hstack([X2, np.ones([N, 1])])
    ])

    random_indexes = np.random.permutation(N).reshape([size, -1])
    random_indexes = np.hstack([random_indexes, random_indexes + N])

    AA = np.stack([AA[random_indexes[idx], :] for idx in range(size)])

comm.Scatter(AA, A, root=0)

MAX_ITER = 120 + 1
ABSTOL = 1e-4
RELTOL = 1e-2
rho, alpha, C = 1, 1.2, 1
objvals, r_norms, s_norms = [], [], []
xvals = []

z, u = np.zeros(M+1), np.zeros(M+1)

STOP_FLAG, i = False, 0

while True:
    if STOP_FLAG or i >= MAX_ITER:
        break

    # Update x
    x_var = cp.Variable(M+1)
    cp.Problem(
        cp.Minimize(cp.sum(cp.pos(cp.matmul(A, x_var)+1)) + rho/2 * cp.sum_squares(x_var-z+u))
    ).solve(verbose=False, solver=cp.CVXOPT)
    x = x_var.value
    x_ave = mpireduce(x, filename="xvals/x{}.npy".format(i))

    # Update z
    z_old = z
    x_hat = alpha*x +(1-alpha) * z_old
    z = rho/(1/C + size*rho) * size * mpireduce(x_hat+u)

    # Update u
    u += (x_hat - z)

    r_norm = mpireduce(x-z, func=max_sigma)
    s_norm = norm(-rho*(z-z_old))
    eps_pri = sqrt(M+1)*ABSTOL + RELTOL*max([norm(x), norm(-z)])
    eps_dual = sqrt(M+1)*ABSTOL + RELTOL*norm(rho*u)

    if rank == 0:
        objval = objective(AA, C, x_ave, z)
        objvals.append(objval)
        s_norms.append(s_norm)
        r_norms.append(r_norm)
        print("{:<3d} | {:<10.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f}".format(
            i, objval, r_norm, eps_pri, s_norm, eps_dual)
        )
        if r_norm < eps_pri and s_norm < eps_dual:
            STOP_FLAG = True

    i += 1
    STOP_FLAG = comm.bcast(STOP_FLAG, root=0)


if rank == 0:
    np.save("tmp/objvals.npy", np.array(objvals))
    np.save("tmp/s_norms.npy", np.array(s_norms))
    np.save("tmp/r_norms.npy", np.array(r_norms))