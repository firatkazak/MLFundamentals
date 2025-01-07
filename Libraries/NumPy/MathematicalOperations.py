import numpy as np

l1 = [1, 2, 3, 4, 5]
l2 = [6, 7, 8, 9, 10]
a1 = np.array(l1)
a2 = np.array(l2)

print(l1 * 5)
# [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]

print(a1 * 5)
# [ 5 10 15 20 25]

print(l1 + l2)
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print(a1 + a2)
# [ 7  9 11 13 15]

b1 = np.array([1, 2, 3])
b2 = np.array([[1], [2]])
print(b1 + b2)
# [[2 3 4]
#  [3 4 5]]
# AÇIKLAMASI;
# b1'in boyutu: (3,) — Bu, 1 satırlık ve 3 sütunluk bir vektördür.
# b2'nin boyutu: (2, 1) — Bu, 2 satırlık ve 1 sütunluk bir matristekidir.
# NumPy, bu iki array'i toplarken boyutları uyumlu hale getirmek için broadcasting uygular:

# b1'i (1, 3) boyutunda bir matrise dönüştürür (satır vektörü haline gelir).
# b2'yi (2, 1)'den (2, 3) boyutunda bir matrise genişletir (her satır aynı değeri alır).

# Sonuç: Bu durumda b1 + b2 şu şekilde hesaplanır:
# [[1, 2, 3]       # b1, her satırda tekrarlandı
#  [1, 2, 3]]
# +
# [[1],            # b2, her sütunda tekrarlandı
#  [2]]
# =
# [[2, 3, 4],
#  [3, 4, 5]]

d = np.array([[1, 2, 3], [4, 5, 6]])
print(np.sqrt(d))  # np.sqrt fonksiyonu, array'deki her bir elemanın karekökünü alır.
# [[1.         1.41421356 1.73205081]
#  [2.         2.23606798 2.44948974]]

f = np.array([[1, 2, 3], [4, 5, 6]])
print(np.sin(f))  # np.sin fonksiyonu, array'deki her bir elemanın sinüsünü alır.
# [[ 0.84147098  0.90929743  0.14112001]
#  [-0.7568025  -0.95892427 -0.2794155 ]]

# NOT: Burada sadece örnek olarak sinüs aldırdık. Sıralı matematiksel işlemlerin listesi;
# https://numpy.org/doc/stable/reference/routines.math.html
