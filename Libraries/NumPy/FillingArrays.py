import numpy as np

a = np.full(shape=(2, 3, 4), fill_value=9)
print(a)  # 2 tane 3 boyutlu, 4 elemanlı dizi veriyor ve içini verdiğimiz sayı ile dolduruyor(9).
# [[[9 9 9 9]
#   [9 9 9 9]
#   [9 9 9 9]]

b = np.zeros(shape=(2, 3, 4))
print(b)  # 2 tane 3 boyutlu, 4 elemanlı dizi verdi ama içi 0 ile dolu şekilde.
# [[[0. 0. 0. 0.]
#   [0. 0. 0. 0.]
#   [0. 0. 0. 0.]]
#
#  [[0. 0. 0. 0.]
#   [0. 0. 0. 0.]
#   [0. 0. 0. 0.]]]

c = np.ones(shape=(2, 3, 4))
print(c)  # 2 tane 3 boyutlu, 4 elemanlı dizi verdi ama içi 1 ile dolu şekilde.
# [[[1. 1. 1. 1.]
#   [1. 1. 1. 1.]
#   [1. 1. 1. 1.]]
#
#  [[1. 1. 1. 1.]
#   [1. 1. 1. 1.]
#   [1. 1. 1. 1.]]]

d = np.empty(shape=(1, 2, 3))  # Belirtilen şekil ve türde yeni bir dizi döndürür.
print(d)
# [[[1.09013873e-311 1.09013873e-311 1.09013872e-311]
#   [1.09013872e-311 1.09013873e-311 1.09013873e-311]]]

x_values1 = np.arange(start=0, stop=101, step=5)
print(x_values1)  # 0'dan 100'e, 5'erli say dedik.
# [  0   5  10  15  20  25  30  35  40  45  50  55  60  65  70  75  80  85 90  95 100]

x_values2 = np.linspace(start=0, stop=100, num=5)
print(x_values2)  # 0'dan 100'e 5 adımda gitti.
# [  0.  25.  50.  75. 100.]
