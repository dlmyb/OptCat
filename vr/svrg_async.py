from mpi4py import MPI
import numpy as np
import random
from collections import Counter
from helper import scatter
import logging

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

r = random.Random()
logging.basicConfig(filename="svrg.log", level=logging.INFO, format='%(asctime)s: P{}: %(message)s'.format(rank))

def df(x, A, b):
    return A.T @ (A @ x - b)

def log_func(func):
    p = func.__name__
    def tmp(*args, **kwargs):
        ret = func(*args, **kwargs)
        # logging.info("{} - {} {}".format(p, args, kwargs))
        logging.info("{} - {}".format(p, list(kwargs.values())))
        return ret
    return tmp

Send = log_func(comm.Send)
Recv = log_func(comm.Recv)

def eval_theta(L, m, lamba, eta, tau):
    t1 = 4*L * (eta + L*tau*tau*eta*eta) / (1-2*L*L*eta*eta*tau*tau)
    print(t1)
    return (1/lamba/eta/m + t1) / (1 - t1)
        
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

MAX_ITER = 500 + 1 # +1 could print info when loop ends.
EPOCH = 8
TAU = 3

if rank == 0:
    lr = 1/L
    print("theta:", eval_theta(L, EPOCH, mu, lr, TAU))
    buf = np.zeros([size-1, M])
    df_new = np.zeros(M)
    x_stored = np.copy(x)
    
    dsts = np.arange(1, size)
    # dists = Counter()
    for i in range(MAX_ITER):
        logging.info("Starting epoch {}".format(i))
        x -= x - x_stored # x = x_stored
        for j in range(size-1):
            comm.send(1, j+1)
            Send(x, dest=j+1)
            Recv(buf[j], source=j+1)
        logging.info("finished sync buf")

        sample = r.randint(a=0, b=EPOCH-1)
        j = 0
        while j < EPOCH:
            dst_round = r.choices(dsts, k=min(TAU, EPOCH-j))
            logging.info("List is {}".format(dst_round))

            # Send a wake up signal
            for dst in dst_round:
                comm.send(1, dst)
                Send(x, dest=dst)

            for _ in dst_round:
                st = MPI.Status()
                comm.probe(status=st)
                dst = st.Get_source()
                df_last = buf[dst-1]
                Recv(df_new, source=dst)
                x -= lr*(df_new - df_last + np.sum(buf, axis=0)/size)
                logging.info("step {} update gradient from {}".format(j, dst))

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
    while comm.recv(source=0):
        Recv(x, source=0)
        Send(df(x, A, b), dest=0)

