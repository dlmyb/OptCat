"""
Classes and functions relating consensus sharing
"""

from mpi4py import MPI
import numpy as np
import matrix

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size


def reduce(target, func=np.mean, filename=None):
    """
    Centralized consensus module based on MPI
    target: numpy array
    func: reduce function
    filename: string, save gathered result
    """
    target_buf, res = None, None
    comm = MPI.COMM_WORLD
    if rank == 0:
        target_buf = np.zeros([size] + list(target.shape))
    comm.Gather(target, target_buf, root = 0)
    if rank == 0 and filename is not None:
        np.save("tmp/" + filename, target_buf)
    if rank == 0:
        res = func(target_buf, axis=0)
    res = comm.bcast(res, root=0)
    return res


class GCon(object):
    """
    Decentralized consensus module based on MPI
    """
    def __init__(self, features, w):
        """
        features: int, dimensions
        w: np.array, each w vector for each worker
        """
        self.x_last = np.zeros(features)
        self.s_last = np.zeros(features)
        self.w = w
        # target and buf are for exchanging data
        self.target = []
        for i, ws in enumerate(w):
            if ws != 0 and i != rank:
                self.target.extend([i])
        self.buf = np.zeros([len(self.target), features])
    
    def __call__(self, x):
        """
        x: np.array
        """
        requests = []
        for i, tar in enumerate(self.target):
            requests.extend([
                comm.Isend(self.s_last * self.w[tar], tar), 
                comm.Irecv(self.buf[i], tar)
            ])
        # Wait all requests finished
        MPI.Request().Waitall(requests)
        # Updating formula, see Shuo Han's CSL letter, formula (3)
        result = np.sum(self.buf, axis=0) + self.w[rank] * \
            self.s_last + (x - self.x_last)
        self.x_last, self.s_last = x, result
        return result