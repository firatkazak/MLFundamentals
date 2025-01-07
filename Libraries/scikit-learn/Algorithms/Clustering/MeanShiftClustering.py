import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift

# Rastgele veri noktaları oluştur
X, _ = make_blobs(n_samples=100, centers=3, random_state=0)

# Mean Shift algoritması ile kümeleme
mean_shift = MeanShift()
labels = mean_shift.fit_predict(X)

# Sonuçları görselleştir
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
plt.title('Mean Shift Kümeleme')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.show()

# Mean Shift Clustering
# Alegori: Düşün ki bir grup insan, en yoğun buluşma noktalarını bulmaya çalışıyor.
# Mean Shift, bu yoğunluk noktalarına doğru kayarak insanları kümeliyor.
