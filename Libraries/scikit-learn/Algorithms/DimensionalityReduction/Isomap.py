import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Iris veri setini yükle
data = load_iris()
X = data.data  # Özellikler

# Veriyi standartlaştır
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Isomap ile 2 boyuta indir (n_neighbors parametresi artırıldı)
isomap = Isomap(n_components=2, n_neighbors=20)  # n_neighbors'ı 20 yapıyoruz
X_isomap = isomap.fit_transform(X_scaled)

# Veriyi görselleştir
plt.scatter(X_isomap[:, 0], X_isomap[:, 1], c=data.target)
plt.title('Isomap ile Boyut İndirgeme')
plt.xlabel('1. Isomap Bileşeni')
plt.ylabel('2. Isomap Bileşeni')
plt.show()

# Isomap
# Alegori: Isomap, bir tur rehberi gibidir.
# Rehber, yüksek dağların ve derin vadilerin arasında en etkili rotayı seçerek,
# turistlerin gizli güzellikleri keşfetmelerini sağlar.
