import matplotlib.pyplot as plt
import umap
from sklearn.datasets import load_iris

# Iris veri setini yükle
data = load_iris()
X = data.data  # Özellikler

# UMAP ile 2 boyuta indir
umap_model = umap.UMAP(n_components=2)
X_umap = umap_model.fit_transform(X)

# Veriyi görselleştir
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=data.target)
plt.title('UMAP ile Boyut İndirgeme')
plt.xlabel('1. UMAP Bileşeni')
plt.ylabel('2. UMAP Bileşeni')
plt.show()

# UMAP(Uniform Manifold Approximation and Projection)
# Alegori: UMAP, bir sanatçının tuvali gibidir.
# Sanatçı, farklı renkleri harmanlayarak en iyi görüntüyü oluşturur;
# bu da verinin en önemli özelliklerini ortaya çıkarır.
