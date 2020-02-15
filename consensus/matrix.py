"""
An submodule relating consensus matrix
"""
import numpy as np

def gen_matrix(n, edges):
    """
    Based on edge set, return a feasible consensus matrix
    n: int, number of nodes
    edges: List[List[int, int]],
    """
    W = np.eye(n)
    degrees = [0] * n
    for i, j in edges:
        degrees[i] += 1
        degrees[j] += 1
    print(degrees)
    for i, j in edges:
        weight = 0.5 / max(degrees[i], degrees[j])
        W[i][j] = weight
        W[j][i] = weight
        W[i][i] -= weight
        W[j][j] -= weight
    return W


def w_feasible(W):
    """
    Check if W is a feasible consensus matrix
    W: consensus matrix
    """
    if W.ndim != 2 or (W.ndim == 2 and W.shape[0] != W.shape[1]):
        return False
    m, _ = W.shape
    # symmetric and stochastic matrix
    return np.all(W == W.T) == True and \
           np.all(np.sum(W, axis=1) == np.ones(m)) == True


def ring_matrix(n):
    """
    return a ring-topology matrix
    n: int, number of nodes
    """
    if n <= 1:
        raise TypeError("n needs to greater than 1")
    w = np.eye(n) * 0.5
    for i in range(n):
        w[i, (i+1)%n] = 0.25
        w[i, (i-1)%n] = 0.25
    return w