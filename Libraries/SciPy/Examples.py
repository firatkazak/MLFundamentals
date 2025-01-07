import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, odeint
from scipy.interpolate import interp1d
from scipy.optimize import minimize


# 1. Soru;
def f(A):
    integrand = lambda t: 2 / (1 + np.abs(A * np.sin(np.pi * t / 10))) ** 2 + 5 * np.sqrt(
        1 + (np.pi * A / 10) ** 2 * np.cos(np.pi * t / 10) ** 2)
    return quad(func=integrand, a=0, b=10)[0]


sonuc = minimize(fun=f, x0=0.001, method="CG").x
print(sonuc)  # [1.42591231] yazar.

# 2. Soru;


t_m = np.array([0., 1.04347826, 2.08695652, 3.13043478, 4.17391304,
                5.2173913, 6.26086957, 7.30434783, 8.34782609, 9.39130435,
                10.43478261, 11.47826087, 12.52173913, 13.56521739, 14.60869565,
                15.65217391, 16.69565217, 17.73913043, 18.7826087, 19.82608696,
                20.86956522, 21.91304348, 22.95652174, 24.])

temp_m = np.array([283.2322975, 284.6945461, 286.2259041, 287.8603625, 289.6440635,
                   291.6187583, 293.7939994, 296.1148895, 298.4395788, 300.5430675,
                   302.1566609, 303.0363609, 303.0363609, 302.1566609, 300.5430675,
                   298.4395788, 296.1148895, 293.7939994, 291.6187583, 289.6440635,
                   287.8603625, 286.2259041, 284.6945461, 283.2322975])

plt.scatter(t_m, temp_m)
plt.xlabel('Time [hour]')
plt.ylabel('Temperature [K]')
Ts = interp1d(t_m, temp_m, kind='cubic')


def dTdt(T, t):
    return -0.5 * (T - Ts(t))


times = np.linspace(start=1, stop=23, num=1000)
T0 = 284.6945461
sol = odeint(dTdt, T0, times).T[0]
plt.plot(times, sol, label='Shallow Water temp')
plt.scatter(t_m, temp_m, color='r', label='Outside Temp')
plt.legend()

plt.show()
