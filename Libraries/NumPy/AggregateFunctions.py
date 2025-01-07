import numpy as np

a = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]])
print(a.min())  # 1 yazar.
print(a.max())  # 20 yazar.
print(a.mean())  # 10.5 yazar.
print(a.std())  # 5.766281297335398 yazar.
print(a.sum())  # 210 yazar.
print(np.median(a))  # 10.5 yazar.
