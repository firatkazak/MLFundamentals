import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.datasets import make_multilabel_classification

# Örnek veri seti oluşturun
X, Y = make_multilabel_classification(n_samples=100, n_features=10, n_classes=3, n_labels=2, random_state=0)

# CCA modelini tanımlayın
cca = CCA(n_components=3)

# Modeli eğitin
cca.fit(X, Y)

# Dönüştürülmüş verileri alın
X_c, Y_c = cca.transform(X, Y)

# İlk iki bileşeni görselleştirin
plt.figure(figsize=(10, 6))
plt.scatter(X_c[:, 0], X_c[:, 1], c='blue', label='X Dönüştürülmüş')
plt.scatter(Y_c[:, 0], Y_c[:, 1], c='red', label='Y Dönüştürülmüş', alpha=0.5)
plt.title('Canonical Correlation Analysis (CCA)')
plt.xlabel('Bileşen 1')
plt.ylabel('Bileşen 2')
plt.legend()
plt.grid()
plt.show()

# CCA korelasyonları
cca_correlation = cca.score(X, Y)
print(f"CCA Korelasyonu: {cca_correlation:.2f}")

# Veri Seti Oluşturma: make_multilabel_classification fonksiyonu ile rastgele bir çok etiketli veri seti oluşturuyoruz. Bu, iki farklı veri seti (X ve Y) sağlar.
# CCA Modeli: CCA sınıfını kullanarak CCA modelini tanımlıyoruz. n_components ile kaç tane kanonik bileşen oluşturmak istediğimizi belirtiyoruz.
# Model Eğitimi ve Dönüştürme: Modeli fit metodu ile eğittikten sonra, transform metodu ile verileri dönüştürüyoruz.
# Görselleştirme: İlk iki bileşeni matplotlib ile görselleştiriyoruz. X'in dönüştürülmüş değerleri mavi, Y'nin dönüştürülmüş değerleri kırmızı ile gösteriliyor.
# Korelasyon Hesabı: CCA korelasyonunu hesaplayıp yazdırıyoruz.
