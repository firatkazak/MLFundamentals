import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Rastgele veri noktaları oluştur
X = np.random.rand(100, 2)

# DBSCAN ile kümeleme
dbscan = DBSCAN(eps=0.1, min_samples=5)
labels = dbscan.fit_predict(X)

# Sonuçları görselleştir
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
plt.title('DBSCAN Kümeleme')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.show()

# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
# Alegori: Düşün ki bir şehirdeki kafeler yoğunluklarına göre gruplandırılıyor.
# DBSCAN, yüksek yoğunluklu bölgeleri (popüler kafeler) belirleyip, gürültü (yeterince müşteri çekmeyen kafeler) olarak adlandırılan alanları ayırıyor.
# Böylece, hangi bölgelerin daha popüler olduğunu anlayabiliyoruz.
