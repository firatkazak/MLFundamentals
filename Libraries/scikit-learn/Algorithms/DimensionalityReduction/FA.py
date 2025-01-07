import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import FactorAnalysis

# Iris veri setini yükle
data = load_iris()
X = data.data  # Özellikler

# Faktör analizi ile 2 boyuta indir
fa = FactorAnalysis(n_components=2)
X_fa = fa.fit_transform(X)

# Veriyi görselleştir
plt.scatter(X_fa[:, 0], X_fa[:, 1], c=data.target)
plt.title('Factor Analysis ile Boyut İndirgeme')
plt.xlabel('1. Faktör')
plt.ylabel('2. Faktör')
plt.show()

# Factor Analysis (FA)
# Alegori: Factor Analysis, bir dedektifin ipuçlarını bir araya getirmesi gibidir.
# Dedektif, olayın arka planını anlamak için en önemli faktörleri bulur.
