import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

# Rastgele veri noktaları oluştur
X, _ = make_blobs(n_samples=100, centers=3, random_state=0)

# GMM ile kümeleme
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
labels = gmm.predict(X)

# Sonuçları görselleştir
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
plt.title('Gaussian Mixture Models Kümeleme')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.show()

# Gaussian Mixture Models (GMM)
# Alegori: Bir ormanda ağaçların türlerini anlamaya çalışıyorsun.
# GMM, her tür ağacın altında farklı dağılımların olduğunu varsayarak bu ağaçları kümeliyor.
