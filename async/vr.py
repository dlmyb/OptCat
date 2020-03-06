from mpi4py import MPI
import numpy as np

def df(x, A, b):
    return A.T.dot(A.dot(x) - b)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

M, batch = 5, 50
A, b = np.zeros([batch, M]), np.zeros([batch])
AA, bb = None, None

if rank == 0:
    AA = np.random.randn(size*batch, M)
    bb = np.random.randn(size*batch, 1)

    # The master gets all zeros
    AA[:batch] = 0
    bb[:batch] = 0

    print("Optimal value:", np.linalg.solve(AA.T @ AA, AA.T @ bb).flatten())
    
    AA = AA.reshape([size, batch, M])
    bb = bb.reshape([size, batch])

comm.Scatter(AA, A, root=0)
comm.Scatter(bb, b, root=0)

mem = np.zeros([M])
if rank == 0:
    mem = np.random.randn(M) * 10
    comm.Bcast(mem, root=0)
win = MPI.Win.Create(mem, comm=comm)

MAX_ITER = 500 + 1
s, eta = 0.1, 1

if rank == 0:
    buf = np.zeros([size-1, M])
    df_new = np.zeros([M])
    x = np.copy(mem)

    requests = [MPI.REQUEST_NULL for i in range(0,size-1)]
    for i in range(size-1):
        requests[i] = comm.Irecv(buf[i], source=i+1)
    MPI.Request.Waitall(requests)

    recv = 0
    for i in range((size-1)*MAX_ITER):
        # Check which worker send its gradient
        while not comm.Iprobe(recv+1):
            recv = (recv+1) % (size-1)
        df_last = buf[recv]
        comm.Recv(df_new, recv+1)
        lr = s * np.sqrt(s / (i + s))

        x = x - lr*(df_new - df_last + np.sum(buf, axis=0)/size)

        win.Lock(0)
        win.Put(x, 0)
        win.Unlock(0)

        buf[recv] = df_new

        # send wake up signal
        comm.send(0, recv+1)

        if i % 200 == 0:
            print(i, x)


else:
    x = np.zeros([M]) + 5
    comm.Send(df(x, A, b), 0)
    for _ in range(MAX_ITER):
        win.Lock(0)
        win.Get(x, 0)
        win.Unlock(0)

        comm.Send(df(x, A, b), 0)
        # Wait master send a wake up signal
        comm.recv(source=0)

