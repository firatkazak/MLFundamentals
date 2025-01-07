import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.datasets import load_iris

# Iris veri setini yükle
data = load_iris()
X = data.data  # Özellikler

# MDS ile 2 boyuta indir
mds = MDS(n_components=2)
X_mds = mds.fit_transform(X)

# Veriyi görselleştir
plt.scatter(X_mds[:, 0], X_mds[:, 1], c=data.target)
plt.title('Principal Coordinates Analysis (MDS) ile Boyut İndirgeme')
plt.xlabel('1. Koordinat')
plt.ylabel('2. Koordinat')
plt.show()

# Principal Coordinates Analysis (PCA)
# Alegori: Principal Coordinates Analysis, bir haritanın tasvirini yapmaya benzer.
# Haritacı, farklı noktaları en uygun şekilde yerleştirerek,
# en iyi yol haritasını sunar.
