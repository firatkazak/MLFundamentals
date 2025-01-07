import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import TruncatedSVD

# Iris veri setini yükle
data = load_iris()
X = data.data  # Özellikler

# SVD ile 2 boyuta indir
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X)

# Veriyi görselleştir
plt.scatter(X_svd[:, 0], X_svd[:, 1], c=data.target)
plt.title('SVD ile Boyut İndirgeme')
plt.xlabel('1. SVD Bileşeni')
plt.ylabel('2. SVD Bileşeni')
plt.show()

# Singular Value Decomposition (SVD)
# Alegori: SVD, bir fotoğrafın çözünürlüğünü azaltmaya benzer.
# Fotoğrafçı, en önemli ayrıntıları koruyarak,
# bir görüntüyü daha hafif hale getirir.
