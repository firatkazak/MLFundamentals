import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import load_iris

# Iris veri setini yükle
data = load_iris()
X = data.data  # Özellikler
y = data.target  # Etiketler

# LDA ile 2 boyuta indir
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X, y)

# Veriyi görselleştir
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=data.target)
plt.title('LDA ile Boyut İndirgeme')
plt.xlabel('1. LDA Bileşeni')
plt.ylabel('2. LDA Bileşeni')
plt.show()

# Linear Discriminant Analysis (LDA)
# Alegori: LDA, bir dedektifin suçluyu bulma yolculuğu gibidir.
# Dedektif, ipuçlarını takip ederek en kritik bilgiyi ortaya çıkarır.
