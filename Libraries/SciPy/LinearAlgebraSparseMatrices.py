import numpy as np
from scipy import sparse
from scipy.linalg import kron  # kronecker product, NOT sum

# 1. Soru;
N = 5
d = -2 * np.ones(N)
e = np.ones(N - 1)
D = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)
D_kronsum = kron(D, np.identity(N)) + kron(np.identity(N), D)
print(D_kronsum)

# 2. Soru;
N = 100
diag = np.ones([N])
diags = np.array([diag, -2 * diag, diag])
D = sparse.spdiags(diags, np.array([-1, 0, 1]), N, N)
T = -1 / 2 * sparse.kronsum(D, D)
print(T)
