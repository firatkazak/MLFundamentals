import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.stats import norm, multinomial, beta

# 1. Soru;
a, b = 2.5, 3.1
mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')
x = np.linspace(beta.ppf(0, a, b), beta.ppf(1, a, b), 100)
plt.plot(x, beta.pdf(x, a, b))
plt.show()  # Resim çıktısına bak.
r = beta.rvs(a, b, size=10)
print(r)  # dizi çıktısına bak.

# 2. Soru;
mu = 1
sigma = 2
mean, var = norm.stats(loc=mu, scale=sigma, moments='mv')
x = np.linspace(norm.ppf(0.01, mu, sigma), norm.ppf(0.99, mu, sigma), 100)
plt.plot(x, norm.pdf(x, mu, sigma))
plt.show()

# 3. Soru;
p = np.ones(6) / 6
sonuc1 = multinomial.pmf([6, 0, 0, 0, 0, 0], n=6, p=p)
sonuc2 = multinomial.rvs(n=100, p=p, size=5)
print(sonuc1)  # 1. Çıktıya bak.
print(sonuc2)  # 2. Çıktıya bak.


# 4. Soru;
class mr_p_solver_dist(st.rv_continuous):
    def _pdf(self, x, a1, a2, b1, b2):
        return 1 / (2 * (a1 * b1 + a2 * b2)) * (b1 * np.exp(-np.sqrt(x / a1)) + b2 * np.exp(-np.sqrt(x / a2)))


my_rv = mr_p_solver_dist(a=0, b=np.inf)

a1, a2, b1, b2 = 2, 3, 1, 2
x = np.linspace(my_rv.ppf(0.01, a1, a2, b1, b2), my_rv.ppf(0.99, a1, a2, b1, b2), 100)
y = my_rv.pdf(x, a1, a2, b1, b2)
plt.plot(x, y)
plt.semilogy()
plt.show()
