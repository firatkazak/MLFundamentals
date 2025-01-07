import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Rastgele veri noktaları oluştur
np.random.seed(0)
X = np.random.rand(100, 2)

# K-Means algoritması ile kümeleri belirle
k = 3
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)

# Küme merkezlerini ve etiketleri al
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Sonuçları görselleştir
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Küme Merkezleri')
plt.title('K-Means Kümeleme')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.legend()
plt.show()

# K-Means Clustering
# Alegori: Düşün ki bir grup insan, belirli özelliklere (örneğin, yaş ve gelir) göre bir etkinlikte bir araya geliyor.
# K-Means, bu insanları benzer özelliklere göre 3 gruba ayırıyor. Her grup, bir merkez (Küme merkezi) etrafında toplanıyor.
# Böylece, organizatörler etkinliklerini daha etkili bir şekilde planlayabiliyor.
