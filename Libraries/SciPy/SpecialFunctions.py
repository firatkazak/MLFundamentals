from scipy.special import legendre
from scipy.special import jv
import matplotlib.pyplot as plt
import numpy as np

# 1. Örnek;
x = np.linspace(start=0, stop=1, num=100)
plt.plot(x, legendre(6)(x))
plt.show()

# 2. Örnek;
x = np.linspace(start=0, stop=10, num=100)
plt.plot(x, jv(3, x))
plt.show()

