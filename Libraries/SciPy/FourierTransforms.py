from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import numpy as np

# 1. Örnek;
x = np.linspace(start=0, stop=10 * np.pi, num=100)
y = np.sin(2 * np.pi * x) + np.sin(4 * np.pi * x) + 0.1 * np.random.randn(len(x))
plt.plot(x, y)
plt.show()

# 2. Örnek;
x = np.linspace(start=0, stop=10 * np.pi, num=100)
y = np.sin(2 * np.pi * x) + np.sin(4 * np.pi * x) + 0.1 * np.random.randn(len(x))
N = len(y)
yf = fft(y)[:N // 2]
xf = fftfreq(N, np.diff(x)[0])[:N // 2]
plt.plot(xf, np.abs(yf))
plt.show()
