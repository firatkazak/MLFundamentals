from scipy.integrate import quad
import numpy as np
from scipy.integrate import dblquad

# Single integrals
integrand = lambda x: x ** 2 * np.sin(2 * x) * np.exp(-x)
integral, integral_error = quad(integrand, 0, 1)
print(integral)  # 0.14558175869954834 yazar.

# Double integrals;
integrand = lambda y, x: np.sin(x + y ** 2)
lwr_y = lambda x: -x
upr_y = lambda x: x ** 2
integral, integral_error = dblquad(integrand, 0, 1, lwr_y, upr_y)
print(integral)  # 0.590090324408853 yazar.
