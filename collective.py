from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def Bcast(buf, root):
    """
    Broadcast buf in root node to all nodes.
    buf: numpy.array
    root: int
    """
    if root != rank:
        comm.Recv(buf, root)
        return
    for target in range(size):
        if target == rank:
            continue
        comm.Send(buf, target)
    

def Scatter(sendbuf, recvbuf, root):
    """
    Scatter a array from root node to all nodes
    sendbuf: numpy.array, dims = [size, M, ...] 
    recvbuf: numpy.array, dims = [M, ...]
    root: int
    """
    if rank != root:
        comm.Recv(recvbuf, root)
        return
    recvbuf = sendbuf[root]
    for target in range(size):
        if target == rank:
            continue
        comm.Send(sendbuf[target], target)


def Gather(sendbuf, recvbuf, root):
    """
    Gather arrays from all nodes to root node
    sendbuf: numpy.array, dims = [M, ...] 
    recvbuf: numpy.array, dims = [size, M, ...]
    root: int
    """
    if rank != root:
        comm.Send(sendbuf, root)
        return
    recvbuf[root] = sendbuf
    for target in range(size):
        if target == rank:
            continue
        comm.Recv(recvbuf[target], target)


def Bcast_nonblocking(buf, root):
    """
    Broadcast buf in root node to all nodes using non-blocking primitives
    buf: numpy.array
    root: int
    """
    if root != rank:
        comm.Recv(buf, root)
        return
    for target in range(size):
        if target == rank:
            continue
        comm.Isend(buf, target)