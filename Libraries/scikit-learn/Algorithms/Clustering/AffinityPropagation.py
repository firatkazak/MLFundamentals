import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AffinityPropagation

# Rastgele veri noktaları oluştur
X, _ = make_blobs(n_samples=100, centers=3, random_state=0)

# Affinity Propagation ile kümeleme
affinity_propagation = AffinityPropagation()
labels = affinity_propagation.fit_predict(X)

# Sonuçları görselleştir
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
plt.title('Affinity Propagation Kümeleme')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.show()

# Affinity Propagation
# Alegori: Farklı sokaklardaki kafeler arasında popülaritesine göre benzerlikler kuruyorsun.
# Affinity Propagation, her kafenin en popüler olduğu noktayı belirliyor ve gruplar oluşturuyor.
