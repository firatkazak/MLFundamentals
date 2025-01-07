import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.datasets import load_iris

# Iris veri setini yükle
data = load_iris()
X = data.data  # Özellikler

# NMF ile 2 boyuta indir
nmf = NMF(n_components=2)
X_nmf = nmf.fit_transform(X)

# Veriyi görselleştir
plt.scatter(X_nmf[:, 0], X_nmf[:, 1], c=data.target)
plt.title('NMF ile Boyut İndirgeme')
plt.xlabel('1. NMF Bileşeni')
plt.ylabel('2. NMF Bileşeni')
plt.show()

# Non-negative Matrix Factorization (NMF)
# Alegori: NMF, bir yemek şefinin tarifleri birleştirmesi gibidir.
# Şef, farklı malzemeleri dikkatlice seçerek,
# lezzetli bir yemek ortaya çıkarır.
