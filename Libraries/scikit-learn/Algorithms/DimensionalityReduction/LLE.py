import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.datasets import load_iris

# Iris veri setini yükle
data = load_iris()
X = data.data  # Özellikler

# LLE ile 2 boyuta indir
lle = LocallyLinearEmbedding(n_components=2)
X_lle = lle.fit_transform(X)

# Veriyi görselleştir
plt.scatter(X_lle[:, 0], X_lle[:, 1], c=data.target)
plt.title('LLE ile Boyut İndirgeme')
plt.xlabel('1. LLE Bileşeni')
plt.ylabel('2. LLE Bileşeni')
plt.show()

# Locally Linear Embedding (LLE)
# Alegori: LLE, bir toplulukta yaşayan insanlar gibidir.
# İnsanlar, komşuları ile etkileşim kurarak en iyi ilişkileri oluşturur;
# bu da topluluğun en önemli yönlerini ortaya çıkarır.
