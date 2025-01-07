import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA

# Iris veri setini yükle
data = load_iris()
X = data.data  # Özellikler

# Feature Agglomeration ile özellikleri birleştir
feature_agglo = FeatureAgglomeration(n_clusters=2)
X_agglo = feature_agglo.fit_transform(X)

# PCA ile boyut indirgeme
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_agglo)

# Veriyi görselleştir
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=data.target)
plt.title('Feature Agglomeration ile Boyut İndirgeme')
plt.xlabel('1. Bileşen')
plt.ylabel('2. Bileşen')
plt.show()

# Alegori: Feature Agglomeration, bir bahçıvanın çiçekleri bir araya getirmesi gibidir.
# Bahçıvan, en güzel çiçekleri bir araya getirerek, harika bir buket oluşturur.
