import numpy as np

# 1. Örnek shape;
a_mul = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(a_mul.shape)  # (3, 4) yazar. 3 tane satır var, her birinin 4 elemanı var.

# 2. Örnek;
a_mul = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]])
print(a_mul.shape)  # (2, 3, 4) yazar. 2 tane dizi var. 3 tane satır var, her birinin 4 elemanı var.

# 3. Örnek;
a_mul = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]])
print(a_mul.ndim)  # 3 yazar. Dizinin boyut sayısını verir.

# 4. Örnek;
a_mul = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]])
print(a_mul.size)  # 24 yazar. Toplam eleman sayısını verir.

# 5. Örnek;
a_mul = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]])
print(a_mul.dtype)  # int64 yazar. Türü verir.
