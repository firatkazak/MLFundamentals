import matplotlib.pyplot as plt
from sklearn.random_projection import GaussianRandomProjection
from sklearn.datasets import load_iris

# Iris veri setini yükle
data = load_iris()
X = data.data  # Özellikler

# Random Projection ile 2 boyuta indir
rp = GaussianRandomProjection(n_components=2)
X_random = rp.fit_transform(X)

# Veriyi görselleştir
plt.scatter(X_random[:, 0], X_random[:, 1], c=data.target)
plt.title('Random Projection ile Boyut İndirgeme')
plt.xlabel('1. Rastgele Projeksiyon Bileşeni')
plt.ylabel('2. Rastgele Projeksiyon Bileşeni')
plt.show()

# Random Projection
# Alegori: Random Projection, bir sanatçının fırça darbeleri gibidir.
# Sanatçı, birçok farklı renkten oluşan bir tabloyu hızlıca oluşturur;
# ancak en önemli unsurları koruyarak.
