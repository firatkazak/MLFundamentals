import numpy as np

# 1. Örnek;

a = np.array([[1, 2, 3], [4, "Hello", 6], [7, 8, 9]])
print(a.dtype)  # <U21 yazar. Data türünü verir.

# 2. Örnek;

a = np.array(object=[[1, 2, 3], [4, "5", 6], [7, 8, 9]], dtype=np.int32)
print(a.dtype)  # int32 yazar. 5 string olmasına rağmen çeviriyor, dikkat et, olay o. Eğer dtype'ını np.int32 vermezsen yine string olurdu.
print(a[1][1].dtype)  # int32 yazar. 5 1. dizinin 1. elemanı. 0 ile başladığı için. yukarıda dtype'ını atadığımız için int32 oluyor.

# 3. Örnek;

d = {"1": "A"}
a = np.array([[1, 2, 3], [4, d, 6], [7, 8, 9]])
print(a.dtype)  # object yazar. Yukarıda d adında bir dictionary yani bir object tanımlayıp diziye ekledik çünkü.

# 4. Örnek;

d = {"1": "A"}
a = np.array([[1, 2, 3], [4, d, 6], [7, 8, 9]], dtype="<U7")
print(a.dtype)  # <U7 yazar. Bu bir numpy türü.

# Numpy data türleri : https://numpy.org/doc/stable/user/basics.types.html
