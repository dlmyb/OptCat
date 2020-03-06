from mpi4py import MPI
import numpy as np
import time
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

SIZE1 = 5
SIZE2 = 10

if rank == 0:
    win =  MPI.Win.Create(None, comm=comm)
    # start remote memory access
    win.Fence()
    res = np.zeros(SIZE1, dtype=np.int64)
    # get the first 5 elements of A at rank 0
    win.Get(res, target_rank=1)
    print('The first 5 elements in rank 1: %s' % res)
    # end remote memory access
    win.Fence()

    # create a group with rank 1 only
    grp = comm.group.Incl(ranks=[1])

    # start remote memory access
    win.Start(grp)
    # put [0,0,0,0,0] in rank 1
    win.Put(np.zeros([5], dtype=np.int64), target_rank=1)
    # end remote memory access
    win.Complete()

    # lock to protect the get operation
    win.Lock(rank=1, lock_type=MPI.LOCK_SHARED)
    # get last 5 elements of A in rank 1 to A[:5] in rank 0
    # Note, MPI.INT = np.int32 = 4, MPI.DOUBLE = np.float = 8, MPI.LONG = np.int64 = 8
    win.Get(res, target_rank=1, target=[5*8, 5, MPI.LONG])
    # unlock after the get operation
    win.Unlock(rank=1)
    print('The last 5 elements in rank 1: %s' % res)
else:
    A = np.zeros(SIZE2, dtype=np.int64) + 1 # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    win =  MPI.Win.Create(A, comm=comm)
    # start remote memory access
    win.Fence()
    # end remote memory access
    win.Fence()

    # create a group with rank 0 only
    grp = comm.group.Incl(ranks=[0])

    # start remote memory access
    win.Post(grp)
    # end remote memory access
    win.Wait()

    # no need for Lock and Unlock here

    print('Finally all elements in rank 1: %s' % A)