import pandas as pd  # Kütüphane Ekleme ve İsim(pd) Verme

df = pd.DataFrame(data=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], columns=["A", "B", "C"], index=["x", "y", "z", "zz"])  # Basit bir DataFrame oluşturma

print(df.head())  # İlk 5 satırı görüntüler.
#      A   B   C
# x    1   2   3
# y    4   5   6
# z    7   8   9
# zz  10  11  12

print(df.tail(2))  # Son 2 satırı görüntüler.
#      A   B   C
# z    7   8   9
# zz  10  11  12

print(df.columns)  # Kolon isimlerini yazdırır.
# Index(['A', 'B', 'C'], dtype='object')

print(df.index)  # DataFrame'in Index'ini verir.
# Index(['x', 'y', 'z', 'zz'], dtype='object')

print(df.info)  # DataFrame hakkında bilgi verir.
# <bound method DataFrame.info of      A   B   C
# x    1   2   3
# y    4   5   6
# z    7   8   9
# zz  10  11  12>

print(df.describe())  # Açıklayıcı istatistiksel özet verir.
#                A          B          C
# count   4.000000   4.000000   4.000000
# mean    5.500000   6.500000   7.500000
# std     3.872983   3.872983   3.872983
# min     1.000000   2.000000   3.000000
# 25%     3.250000   4.250000   5.250000
# 50%     5.500000   6.500000   7.500000
# 75%     7.750000   8.750000   9.750000
# max    10.000000  11.000000  12.000000

print(df.nunique())  # Belirtilen kolondaki farklı öğelerin sayısını sayar. Tekrar edenleri saymıyor yani.
# A    4
# B    4
# C    4
# dtype: int64

print(df['A'].unique())  # Bir sütundaki benzersiz değerlere erişim sağlar.
# [ 1  4  7 10]

print(df.shape)  # DataFrame'in yapısını verir.
# (4, 3)

print(df.size)  # DataFrame'in boyutunu verir.
# 12
