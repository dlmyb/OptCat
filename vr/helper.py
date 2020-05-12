from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


"""
You could use scatter function substitute built-in comm.scatter
It support heterogeneous data scattering
dsts is the portion,
e.g. dsts = {1: 1, 2: 1, 3: 2} -> scatter one part data to node 1 and 2, two parts to node 3.
when dsts is None, all nodes will get the same dimension data.
"""
def scatter(sendbuf, dsts=None, root=0):
    if dsts is None:
        dsts_round = np.ones(size, np.int)
        dsts_round_sum = size
    elif isinstance(dsts, dict):
        dsts_round = np.zeros(size, np.int)
        dsts_round_sum = 0
        for i, v in dsts.items():
            dsts_round[i] = v
            dsts_round_sum += v
    elif isinstance(dsts, np.ndarray):
        """
        TODO: verify dsts
        """
        dsts_round_sum = np.sum(dsts)
        dsts_round = dsts
    else:
        raise TypeError("dsts must be dict/ndarray/None.")
    if rank == root:
        cycle = comm.bcast(len(sendbuf)//dsts_round_sum, root=root)
        shape = comm.bcast(sendbuf.shape[1:], root=root)
        dsts_round *= cycle
        last = 0
        for dst, v in enumerate(np.cumsum(dsts_round)):
            if v == last:
                continue
            comm.Isend(sendbuf[last:v].flatten(), dst, 0xff)
            last = v
    else:
        cycle = comm.bcast(1, root=root)
        shape = comm.bcast(1, root=root)
        dsts_round *= cycle
    if dsts_round[rank] == 0:
        return np.array([])
    buf_shape = [dsts_round[rank]]
    buf_shape.extend(shape)
    buf = np.zeros(buf_shape).flatten()
    comm.Recv(buf, root, 0xff)
    buf = buf.reshape(buf_shape)
    return buf