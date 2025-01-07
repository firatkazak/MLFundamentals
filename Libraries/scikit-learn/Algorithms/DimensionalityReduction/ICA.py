import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.datasets import load_iris

# Iris veri setini yükle
data = load_iris()
X = data.data  # Özellikler

# ICA ile 2 boyuta indir
ica = FastICA(n_components=2)
X_ica = ica.fit_transform(X)

# Veriyi görselleştir
plt.scatter(X_ica[:, 0], X_ica[:, 1])
plt.title('ICA ile Boyut İndirgeme')
plt.xlabel('1. ICA Bileşeni')
plt.ylabel('2. ICA Bileşeni')
plt.show()

# Independent Component Analysis (ICA)
# Alegori: ICA, bir grup müzisyenin senfonisi gibidir.
# Her müzisyen, kendi enstrümanında bağımsız bir şekilde çalar,
# ancak birlikte en güzel melodiyi oluştururlar.
