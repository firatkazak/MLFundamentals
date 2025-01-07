import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from pygam import LinearGAM, s
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Örnek veri seti oluşturun
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=0)

# Veriyi eğitim ve test olarak ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MARS modelini tanımlayın
gam = LinearGAM(s(0)).fit(X_train, y_train)

# Tahmin yapın
y_pred = gam.predict(X_test)

# Sonuçları görselleştirin
plt.scatter(X, y, color='blue', label='Gerçek Veri')
plt.scatter(X_test, y_pred, color='red', label='MARS Tahminleri')
plt.title('Multivariate Adaptive Regression Splines (MARS) with pyGAM')
plt.xlabel('Özellikler')
plt.ylabel('Hedef Değerler')
plt.legend()
plt.grid()
plt.show()

# Mean Squared Error (MSE) hesaplayın
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Multivariate Adaptive Regression Splines (MARS)
# Açıklama
# Veri Seti Oluşturma: make_regression fonksiyonu ile bir regresyon veri seti oluşturuyoruz.
# Veriyi Eğitim ve Test Olarak Ayırma: train_test_split ile verileri eğitim ve test setlerine ayırıyoruz.
# MARS Modeli Tanımlama: LinearGAM sınıfını kullanarak MARS modelini tanımlıyoruz. s(0) ile ilk özelliği kullanıyoruz.
# Tahmin Yapma: Model ile test verileri üzerinde tahmin yapıyoruz.
# Görselleştirme: Gerçek verileri ve tahmin edilen değerleri görselleştiriyoruz.
# Hata Hesabı: Mean Squared Error (MSE) hesaplayarak modelin başarısını değerlendiriyoruz.
