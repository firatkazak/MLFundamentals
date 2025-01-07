import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import SpectralClustering

# Rastgele veri noktaları oluştur
X, _ = make_blobs(n_samples=100, centers=3, random_state=0)

# Spektral kümeleme
spectral_clustering = SpectralClustering(n_clusters=3, affinity='nearest_neighbors')
labels = spectral_clustering.fit_predict(X)

# Sonuçları görselleştir
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
plt.title('Spektral Kümeleme')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.show()

# Spectral Clustering
# Alegori: Düşün ki farklı müzik türlerinin karmaşık bir yapısı var. Spektral kümeleme, bu müzik türlerini benzerliklerine göre ayırıyor.
