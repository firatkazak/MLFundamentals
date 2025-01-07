import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(type(a))  # Tür artık Numpy Array oldu.
# <class 'numpy.ndarray'>

b = np.array([1, 2, 3, 4, 5])
b[2] = 10
print(b)  # 2. eleman olan 3 yerine 10 eklendi.
# [ 1  2 10  4  5]

c_mul = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(c_mul[0])  # [1 2 3] yazar. 3 elemanlı array'in 1. dizisi 1,2 ve 3 rakamlarından oluşuyor. Onu aldık.
# [1 2 3]
print(c_mul[0, 1])  # 2 yazar. 0. array'in 1. elemanı 2.
# 2
