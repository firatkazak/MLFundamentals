import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import Birch

# Rastgele veri noktaları oluştur
X, _ = make_blobs(n_samples=100, centers=3, random_state=0)

# BIRCH ile kümeleme
birch = Birch(n_clusters=3)
labels = birch.fit_predict(X)

# Sonuçları görselleştir
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
plt.title('BIRCH Kümeleme')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.show()

# BIRCH
# Alegori: Büyük bir kütüphanede farklı türde kitaplar var. BIRCH, kitapları türlerine göre düzenli bir şekilde grupluyor.
