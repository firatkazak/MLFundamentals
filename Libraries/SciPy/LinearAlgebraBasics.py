import numpy as np
from scipy.linalg import solve_toeplitz, solve_triangular, cholesky, eigh_tridiagonal, fiedler, toeplitz, lu

# Triangular matrices;
a = np.array([[3, 0, 0, 0],
              [2, 1, 0, 0],
              [1, 0, 1, 0],
              [1, 1, 1, 1]])

b = np.array([4, 2, 4, 2])
x = solve_triangular(a, b, lower=True)
print(x)  # [ 1.33333333 -0.66666667  2.66666667 -1.33333333] yazar.

# Toeplitz Matrices(matrices with constant diagonals);
c = np.array([1, 3, 6, 10])  # First column of T
r = np.array([1, -1, -2, -3])  # First row of T
b = np.array([1, 2, 2, 5])
x = solve_toeplitz(c_or_cr=(c, r), b=b)
print(x)  # [ 1.66666667 -1.         -2.66666667  2.33333333] yazar.

# Eigenvalue Problems;
d = 3 * np.ones(4)
e = -1 * np.ones(3)
w, v = eigh_tridiagonal(d, e)
A = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)
print(A)  # 1. diziye bak.
print(A @ v.T[0])  # 2. diziye bak.
print(w[0] * v.T[0])  # 3. diziye bak.

# Special Matrices;
print(fiedler([1, 4, 12, 45]))  # Çıktıdaki 1. dizeye bak.
print(toeplitz([1, 2, 3, 6, 0, 0], [1, 4, 5, 6, 0, 0]))  # Çıktıdaki 2. dizeye bak.

# Decompositions;
A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
p, l, u = lu(A)
A = np.array([[1, 0.2], [0.2, 1]])
C = cholesky(A, lower=True)
print(C)  # Çıktıdaki 1. dizeye bak.
print(C @ C.T)  # Çıktıdaki 2. dizeye bak.
print(A)  # Çıktıdaki 3. dizeye bak.
