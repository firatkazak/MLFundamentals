import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris

# Iris veri setini yükle
data = load_iris()
X = data.data  # Özellikler

# t-SNE ile 2 boyuta indir
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X)

# Veriyi görselleştir
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=data.target)
plt.title('t-SNE ile Boyut İndirgeme')
plt.xlabel('1. t-SNE Bileşeni')
plt.ylabel('2. t-SNE Bileşeni')
plt.show()

# t-Distributed Stochastic Neighbor Embedding (t-SNE)

# Alegori: t-SNE, bir ormanda kaybolmuş bir gezgin gibidir.
# Gezgin, yüksek ağaçların arasındaki gizli yolları keşfederken,
# ormanın en güzel manzaralarını korur.
