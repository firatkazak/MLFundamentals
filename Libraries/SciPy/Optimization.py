import numpy as np
from scipy.optimize import minimize


# 1. Soru;

def f(x):
    return (x - 3) ** 2


# x0 parametresini bir Numpy array olarak veriyoruz
res = minimize(f, x0=np.array([2]))  # x0 artık bir ndarray

print("Çözüm: ", res.x)  # Optimum çözüm noktası Çözüm:  [2.99999999]

# 2. Soru
f = lambda x: (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2

cons = ({'type': 'ineq', 'fun': lambda x: x[0] - 2 * x[1] + 2},
        {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
        {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})

bnds = ((0, None), (0, None))

# Resme bak. x'in olduğu yerde 2.0 var. Onu buraya verdik.
result = minimize(fun=f, x0=(2, 0), bounds=bnds, constraints=cons)
print(result.x)  # x: 1.4 y: 1.7 çıktı sonuç
