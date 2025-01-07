import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Iris veri setini yükle
data = load_iris()
X = data.data  # Özellikler

# PCA ile 2 boyuta indir
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Veriyi görselleştir
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=data.target)
plt.title('PCA ile Boyut İndirgeme')
plt.xlabel('1. Principal Component')
plt.ylabel('2. Principal Component')
plt.show()

# PrincipalComponentAnalysis(PCA)
# Alegori: PCA, yüksek dağların üzerinden geçen bir yollar sistemi gibidir.
# Yol, en yüksek noktalar arasında en kısa ve düz şekilde uzanırken,
# dağlar arasındaki en önemli bilgileri korur.
