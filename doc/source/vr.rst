==================
Variance Reduction
==================

Variance Reduction (VR) reduced noise while estimating gradient in SGD. Here's a brief `slide <https://github.com/dlmyb/OptCat/blob/master/doc/presentation.pdf>`_.

We implemented SVRG, SAGA on MPI, see the vr folder.

------------
Asynchronous
------------

See `svrg_async.py <https://github.com/dlmyb/OptCat/blob/master/vr/svrg_async.py>`_ , yet another MPI implementation mentioned in `[Reddi et. al.] <http://arxiv.org/abs/1506.06840>`_.