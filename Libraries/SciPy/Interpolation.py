import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Soru 1: Aşağıdaki verilere sahip olduğunuzu varsayalım; (Bunları burada ben oluşturuyorum, ancak belki siz x ve y'yi bir deney yoluyla topluyorsunuz)
x = np.linspace(start=0, stop=10, num=10)
y = x ** 2 * np.sin(x)
plt.scatter(x, y)
plt.show()

# Soru 2: Aradaki değerleri bilmek istiyoruz;
x = np.linspace(start=0, stop=10, num=10)
y = x ** 2 * np.sin(x)
f = interp1d(x, y, kind='cubic')
x_dense = np.linspace(start=0, stop=10, num=100)
y_dense = f(x_dense)
plt.plot(x_dense, y_dense)
plt.scatter(x, y)
plt.show()
