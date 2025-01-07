import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Rastgele veri noktaları oluştur
X, _ = make_blobs(n_samples=100, centers=3, random_state=0)

# Hiyerarşik kümeleme
hierarchical = AgglomerativeClustering(n_clusters=3)
labels = hierarchical.fit_predict(X)

# Dendrogram oluştur
plt.figure(figsize=(10, 5))
Z = linkage(X, method='ward')
dendrogram(Z)
plt.title('Hiyerarşik Kümeleme Dendrogramı')
plt.xlabel('Numara')
plt.ylabel('Mesafe')
plt.show()

# Sonuçları görselleştir
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
plt.title('Hiyerarşik Kümeleme')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.show()

# Hierarchical Clustering
# Alegori: Hayal et ki bir okulda öğrenciler, benzer ilgi alanlarına göre gruplara ayrılıyor.
# Hiyerarşik kümeleme, bu öğrencileri önce daha geniş gruplara ayırıyor(tüm sanatçılar bir arada),
# ve ardından daha spesifik gruplara (resim yapanlar, müzikle ilgilenenler) ayırıyor.
# Böylece, hangi öğrencilerin hangi aktivitelerde daha çok ilgisi olduğunu daha iyi anlayabiliyoruz.
