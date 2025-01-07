import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.datasets import load_iris

# Iris veri setini yükle
data = load_iris()
X = data.data  # Özellikler

# MDS ile 2 boyuta indir
mds = MDS(n_components=2, random_state=0)
X_mds = mds.fit_transform(X)

# Veriyi görselleştir
plt.scatter(X_mds[:, 0], X_mds[:, 1], c=data.target)
plt.title('MDS ile Boyut İndirgeme')
plt.xlabel('1. MDS Bileşeni')
plt.ylabel('2. MDS Bileşeni')
plt.show()

# Multi-dimensional Scaling (MDS)
# Alegori: MDS, bir haritacı gibi düşünebiliriz.
# Haritacı, dağlar ve vadiler arasında en doğru yolları çizer;
# böylece insanlara en uygun güzergahı bulmalarında yardımcı olur.
