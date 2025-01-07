import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# Örnek veri kümesi oluşturma
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# KMeans kümeleme
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Sonuçları görselleştirme
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('KMeans Clustering')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.show()

# Subspace clustering, veri kümesindeki farklı alt alanlarda kümeler bulmaya yönelik bir tekniktir.
# Yukarıda, Python'da sklearn kütüphanesini kullanarak basit bir subspace clustering örneği verilmiştir.
# Bu örnek, KMeans algoritmasını kullanarak alt alanlarda kümeler oluşturacaktır.

# AÇIKLAMALAR
# Veri Kümesi: make_blobs fonksiyonu, belirli sayıda küme merkezine sahip rastgele bir veri seti oluşturur.
# KMeans: KMeans algoritması, veri noktalarını belirtilen sayıda kümeye ayırır. Burada 3 küme kullanılmıştır.
# Görselleştirme: Kümeleme sonuçları ve merkezleri görselleştirilmiştir.

# Notlar
# Eğer alt alanlar oluşturmak istiyorsanız, veri kümenizdeki özellikleri (features) belirli kombinasyonlar ile filtrelemeniz gerekebilir.
# Örneğin, belirli özellikleri kullanarak alt alanlarda KMeans uygulayabilirsiniz.
