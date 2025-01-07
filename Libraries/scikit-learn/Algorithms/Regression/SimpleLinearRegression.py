import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Veri: Deneyim (yıl) ve Maaş ($)
X = np.array([[1], [2], [3], [4], [5], [6]])  # Bağımsız değişken: deneyim
y = np.array([30000, 35000, 40000, 45000, 50000, 60000])  # Bağımlı değişken: maaş

# Model oluşturma;
model = LinearRegression()  # LinearRegression sınıfından bir model nesnesi oluştururuz.
model.fit(X, y)  # Modeli verilerle eğitiriz. Yani, X (deneyim) ve y (maaş) verilerine göre doğrusal bir ilişki öğrenilir.
# NOT: ÖNCE X: training data SONRA y: Target value olacak. Unutma.

# Tahmin yapma
y_pred = model.predict(X)  # Model, verilen X verisi için maaş tahminleri yapar.
# Bu, modelin eğitildiği verilere dayanarak her bir deneyim yılı için tahmin ettiği maaş değerlerini verir.
# Bu tahminler y_pred olarak kaydedilir.

# Sonuçları görselleştirme
# Gerçek verileri (deneyim ve maaş) bir dağılım grafiği (scatter plot) olarak çizeriz.
# Mavi noktalar, her bir deneyim yılı için gerçek maaşları gösterir.
plt.scatter(X, y, color='blue', label='Gerçek Maaşlar')

# Modelin tahmin ettiği maaş değerlerini kırmızı bir çizgiyle çizeriz.
# Bu çizgi, modelin tahmin ettiği doğrusal ilişkiyi temsil eder.
plt.plot(X, y_pred, color='red', label='Tahmini Maaşlar')

plt.xlabel('Deneyim (yıl)')  # X ekseninin etiketini "Deneyim (yıl)" olarak ayarlar.
plt.ylabel('Maaş ($)')  # Y ekseninin etiketini "Maaş ($)" olarak ayarlar.
plt.title('Simple Linear Regression')  # Grafiğe Simple Linear Regression isminde başlık ekler.
plt.legend()  # Etiketler (Gerçek ve Tahmini Maaşlar) için bir açıklama (legend) ekler. Resmin içindeki açıklama kısmı.
plt.show()  # Grafiği gösterir.

# Simple Linear Regression
# Bu regresyon türü, tek bir bağımsız değişken ile bağımlı değişken arasındaki doğrusal ilişkiyi modellemek için kullanılır.
