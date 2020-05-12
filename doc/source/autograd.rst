========================
Tips AutoGrad on Python
========================

.. toctree::
   :maxdepth: 2

For reference, you could read "`Automatic Differentiationin Machine Learning: a Survey <https://arxiv.org/pdf/1502.05767.pdf>`_".

On Python, there are two libraries you could try, PyTorch and autograd (Until now, the core contributors already archieve this repo and move to JAX published by Google). 

-------
PyTorch
-------

For gradient descent, assume we want to minimize 

.. math :: \sum_{a_i \in A}||a_i - x||^2

the example code is follow:

.. code-block :: python
    :linenos:

    import os
    import torch
    import torch.distributed as dist
    from torch.multiprocessing import Process

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    def init_process(rank, size, fn, backend='gloo'):
        """ Initialize the distributed environment. """
        dist.init_process_group(backend, rank=rank, world_size=size)
        fn(rank, size)

    def start_parallel(func, n = 4, **kwargs):
        processes = []
        for rank in range(n):
            p = Process(target=init_process, args=(rank, n, func))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


    MAX_ITER, k, lr = 30, 50, 0.1
    A = torch.normal(0, 1, [k, 2], requires_grad=False)
    x = torch.tensor([100.0, 50.0], requires_grad=True) # x is an arbitrary vector. 

    for i in range(MAX_ITER):
        u = torch.pow(A - x, 2)
        y = torch.sum(u)

        # here's important, whether or not it's master or worker
        # you need to initialize gather_result in each node
        # but only master will receive all results
        # Similar with MPI, you could read MPI example code
        gather_result = []

        if rank == 0:
            gather_result = [torch.zeros(y.size()) for _ in range(size)]
        dist.gather(y, gather_result)

        # get the gradient
        y.backward()

        # boardcast avg gradient
        dx = x.grad.clone() / k
        dist.all_reduce(dx, dist.ReduceOp.SUM)
        
        if rank == 0 and i % 5 == 0:
            # x | dx | \frac{\sum f_i(x)}{k*size}
            print(i, "--", x.data, "--", dx, "--", sum(gather_result)/k/size)

        # update x
        x.data.sub_(lr, dx.data)

        # clear gradient record, it's necessary.
        # AGAIN, IT'S NECESSARY
        x.grad.detach_()
        x.grad.zero_()

Although the code showing above covers distributed part, it's also a demo how to use PyTorch distributed library, you could read corresponding documents, or read MPI books

Useful materials include PyTorch document, `autograd` section and `torch.optim` section. 

----
JAX
----

Similar code is here.

.. code-block :: python
    :linenos:

    from jax import grad
    import jax.numpy as np # Note, it's not NUMPY
    import numpy as onp # Here's numpy

    # define ||A-x||^2
    def f(x, A):
        return np.sum(np.square(A - x))

    # Grad for f
    def df(x, A):
        return grad(f)(x, A)

    MAX_ITER, k, lr = 30, 50, 0.1
    A = np.array(onp.random.rand(k, 2))
    x = np.zeros(2)

    for iter in range(MAX_ITER):
        dx = df(x, A)
        if iter % 5 == 0:
            # iter | x | dx | f
            print(iter, x, dx, f(x, A))
        x -= lr * df(x, A)

You could read docunment for more details, and you could use `jit` to speed up, e.g ``df = jit(grad(jit(f))(x, A)``.