import matplotlib.pyplot as plt
import numpy as np
from findiff import FinDiff


def f(x):
    return x ** 2 * np.sin(2 * x) * np.exp(-x)


# x aralığını tanımlıyoruz
x = np.linspace(start=0, stop=1, num=100)

# f(x) fonksiyonunu hesaplıyoruz
y = f(x)

# İlk türevi hesaplıyoruz
d_dx = FinDiff(0, x[1] - x[0], 1)
first_derivative = d_dx(y)

# İkinci türevi hesaplıyoruz
d2_dx2 = FinDiff(0, x[1] - x[0], 2)
second_derivative = d2_dx2(y)

# Grafikleri çiziyoruz
plt.plot(x, y, label='f(x)')
plt.plot(x, first_derivative, label='f\'(x)')
plt.plot(x, second_derivative, label='f\'\'(x)')
plt.grid()
plt.legend()
plt.show()
