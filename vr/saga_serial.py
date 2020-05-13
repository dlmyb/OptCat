import numpy as np

N, BATCH, M = 8, 50, 2

def df(x, A, b):
    return A.T @ (A @ x - b)

np.random.seed(0)
AA = np.random.randn(N*BATCH, M)
bb = np.random.randn(N*BATCH)
print("Optimal value:", np.linalg.solve(AA.T @ AA, AA.T @ bb))

AA = AA.reshape([N, BATCH, M])
bb = bb.reshape([N, BATCH])

df_buf = np.zeros([N, M])
x = np.zeros([M])
for i in range(N):
    df_buf[i] = df(x, AA[i], bb[i])
df_sum = np.sum(df_buf, axis=0)

i, MAX_ITER, lr = 0, 500+1, 0.001

STOP_FLAGS = np.zeros([N], dtype=np.bool)

while True:
    if i >= MAX_ITER or np.all(STOP_FLAGS == True):
        print(i, x)
        break

    STOP_FLAGS &= False

    for j in np.random.permutation(np.arange(N)):
        df_last, df_cur = df_buf[j], df(x, AA[j], bb[j])
        g = df_cur - df_last + df_sum / N
        x -= lr*g
        df_sum -= df_last - df_cur
        df_buf[j] = df_cur
        if np.linalg.norm(g) <= 1e-6:
            STOP_FLAGS[j] = True

    i += 1


