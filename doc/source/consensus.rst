=========================
Consensus Optimization 
=========================

----------------
Consensus Matrix
----------------

.. py:module:: consensus.matrix
.. py:function:: gen_matrix(n, edges)

    See `Eq4 in [Na Li et al] <https://nali.seas.harvard.edu/files/nali/files/2017tcns_smoothness_dist_opt.pdf>`_.
    
    Based on edge set of graph, return a feasible consensus matrix.

    :param int n: number of vertex
    :param List[List[int]] edges: The list for edge pair
    :return: consensus matrix
    :rtype: numpy.array


.. py:function:: w_feasible(W)

    Check if W is a feasible consensus matrix.
    The consenus should satisfied following constraints:

    .. math ::
        W = W^T \quad \mathcal{1}^T W = \mathcal{1}^T

    :param numpy.array W: consensus matrix
    :rtype: bool

.. py:function:: ring_matrix(n)

    return a ring-topology matrix

    :param int n: int, number of nodes
    :return: consensus matrix
    :rtype: numpy.array


--------------------
Consensus Algorithms
--------------------

In `[Shuo's paper] <https://hanshuo.people.uic.edu/papers/han2019sdd.pdf>`_ , it mentioned two type of algorithms for solve the consensus optimization problem.
Here we take least-square as a example,

.. math :: 
    \min \quad \sum_{i=1}^N{||A_i x + b_i||^2_2} \\
    A_i \in \mathbb{R}^{m_i \times n} \quad b_i \in \mathbb{R}^{m_i}

We let :math:`f_i(x_i) = ||A_i x_i + b_i||^2_2`, therefore we have,

.. math ::
    \min \quad &\sum_{i=1}^N f_i(x_i) \\
    & x_i - z = 0

Solving by ADMM, we have 

.. math ::
    x^{k+1}_i &= (A_i^T A_i + \rho \cdot I)^{-1} (A_i^T b + \bar{x}^k - y^k_i) \\ 
    y^{k+1}_i &= y_i^k + \rho \cdot (x^{k+1} - \bar{x}^{k+1})

We build a `consensus` module for solving consensus optimization under MPI platform, you could use following methods to achieve :math:`\bar{x}`.

^^^^^^^^^^^^^^^^^^^^^
Centralized Algorithm
^^^^^^^^^^^^^^^^^^^^^

In our consensus library, `mpireduce` is similar with average function.

.. py:module:: consensus
.. py:function:: mpireduce(target, func=np.mean, filename=None)

    Centralized consensus module based on MPI, similar with reduce op on MPI.
    You could define your `func` as a reduce function, such as doing SVD on the gather matrix, 
    then boardcast the maximum sigma to every agents.
    The `filename` needs to define if you want to storage the gather matrix, it will be placed under "./tmp" folder

    :param numpy.array target: the consensus object
    :param function func: reduce op
    :param str filename: the filename
    :return: average result
    :rtype: numpy.array

Example code is at `admm.py <https://github.com/dlmyb/OptCat/blob/master/admm.py>`_.


^^^^^^^^^^^^^^^^^^^^^^^
Decentralized Algorithm
^^^^^^^^^^^^^^^^^^^^^^^

Given a consensus matrix :math:`W`, we need to build a `GCon` for each variable which needs to average.


.. py:class:: GCon

    Decentralized consensus module based on MPI

    .. py:method:: __init__(self, features, w)


        :param int features: dimention
        :param numpy.array w: consensus matrix

    .. py:method:: __call__(self, x)

        :param numpy.array x: a `features` dimention vector
        :return: average result
        :rtype: numpy.array

Example code is at `admm_dc.py <https://github.com/dlmyb/OptCat/blob/master/admm_dc.py>`_, we also provide a gradient descent version `gd_dc.py <https://github.com/dlmyb/OptCat/blob/master/gd_dc.py>`_.

------------
Application
------------

^^^^^^^^^^^^
Parallel SVM
^^^^^^^^^^^^

We followed `[Boyd et al] <https://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf>`_ and implement corresponding algorithm on Python.

Giving the :math:`X \in \mathbb{R}^{m_i \times n}, y \in \{+1, -1\}^{m_i}, \lambda \in \mathbb{R}`, the SVM optimization problem is following:

.. math ::
   \min \quad \frac{1}{2 \lambda} || \omega ||^2_2 + \sum_{i=1}^{m_i} \biggl[ 1 - y_i(\omega^T x_i+b)\biggr]^+

We slightly modify the L2-regularizer term to :math:`||[w, b]^T||^2_2`. Let 

.. math ::
   f = \bigl[ \omega, b \bigr]^T, A_i \in \mathbb{R}^{m_i \times (n+1)}, (A_i)_j = \bigl[ -y_j \cdot x_j, -y_j \bigr] 

Therefore, we would have the following problem:

.. math ::
   \min \quad &\sum_{i=1}^N \mathbb{1}^T \bigl[ 1 + A_i \cdot f_i \bigr]^+ + \frac{1}{2 \lambda} || z ||^2_2 \\
   & s.t \quad f_i - z = 0

Solving by ADMM algorithm, we have updates on `[p66 8.2.3] <https://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf>`_.

.. warning :: 
   The currect z-update is :math:`z^{k+1} := \frac{N \rho}{(1/ \lambda) + N \rho}(\bar{x}^{k+1}+\bar{u}^k)`.


"""""""""""
Centralized
"""""""""""

See example code `svm.py <https://github.com/dlmyb/OptCat/blob/master/svm.py>`_.

"""""""""""""
Decentralized
"""""""""""""

See example code `svm_dc.py <https://github.com/dlmyb/OptCat/blob/master/svm_dc.py>`_, Details are  `svm_view.ipynb <https://github.com/dlmyb/OptCat/blob/master/svm_view.ipynb>`_.