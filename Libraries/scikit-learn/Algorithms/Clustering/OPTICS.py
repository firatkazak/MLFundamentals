import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import OPTICS

# Rastgele veri noktaları oluştur
X, _ = make_blobs(n_samples=100, centers=3, random_state=0)

# OPTICS ile kümeleme
optics = OPTICS(min_samples=5, xi=.05, min_cluster_size=.05)
labels = optics.fit_predict(X)

# Sonuçları görselleştir
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
plt.title('OPTICS Kümeleme')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.show()

# OPTICS
# Alegori: Bir şehirdeki farklı mekanların yoğunluğunu anlamaya çalışıyorsun.
# OPTICS, yoğun alanları belirleyerek hangi mekanların daha popüler olduğunu keşfediyor.
