import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Veri: Deneyim (yıl) ve Maaş ($)
X = np.array([[1], [2], [3], [4], [5], [6]])
# X, bağımsız değişkenleri temsil eder; burada her bir gözlem için deneyim (yıl) değeri içerir.

y = np.array([30000, 35000, 40000, 45000, 50000, 60000])
# y, bağımlı değişkeni (Maaş) temsil eder; her bir gözlem için maaş değerleri içerir.

# Polinom dönüşümü
poly = PolynomialFeatures(degree=2)  # Polinomal dönüşüm oluşturuyoruz; degree=2, ikinci derece polinom anlamına gelir.
X_poly = poly.fit_transform(X)  # X verilerini polinomal hale getiriyoruz; X_poly, polinomal özellikler içerir.

# Model oluşturma
model = LinearRegression()  # Lineer regresyon modelini oluşturuyoruz.
model.fit(X_poly, y)  # Modeli polinomal verilerle eğitiyoruz; X_poly bağımsız değişkenler, y ise bağımlı değişkendir.

# Tahmin yapma
y_pred = model.predict(X_poly)  # Model ile polinomal veriler için tahminler yapıyoruz; sonuçları y_pred değişkenine atıyoruz.

# Sonuçları görselleştirme;
plt.scatter(X, y, color='blue', label='Actual Salaries')  # Gerçek maaş verilerini mavi renkte noktalarla çiziyoruz.
plt.plot(X, y_pred, color='red', label='Polynomial Regression Curve')  # Tahmin edilen polinomal regresyon eğrisini kırmızı renkte çiziyoruz.
plt.xlabel('Experience (years)')  # X ekseni için etiket
plt.ylabel('Salary ($)')  # Y ekseni için etiket
plt.title('Polynomial Regression')  # Grafiğin başlığını belirtiyoruz.
plt.legend()  # Grafikteki etiketler için bir efsane (legend) oluşturuyoruz.
plt.show()  # Grafiği gösteriyoruz.

# Polynomial Regression
# Bağımsız ve bağımlı değişken arasındaki doğrusal olmayan (eğrisel) ilişkileri modellemek için kullanılır.

# Genel Açıklama: Bu kod, deneyim ve maaş arasındaki ilişkiyi polinomal regresyon kullanarak modellemektedir.

# Veri Tanımlama: İlk olarak, bağımsız değişken olarak deneyim (yıl) ve bağımlı değişken olarak maaş ($) için veri kümesi tanımlanır.

# Polinomal Dönüşüm: PolynomialFeatures sınıfı kullanılarak deneyim verisi ikinci dereceden bir polinoma dönüştürülür.
# Bu, modelin daha karmaşık bir ilişkiyi öğrenmesine olanak tanır.

# Model Oluşturma ve Eğitme: Bir lineer regresyon modeli oluşturulur ve polinomal verilerle eğitilir.

# Tahmin Yapma: Model, eğitim verileri üzerinde tahminler yapar.

# Görselleştirme: Gerçek maaş verileri ve tahmin edilen polinomal regresyon eğrisi bir grafikte görselleştirilir.

# Sonuç: Bu kod, bağımsız değişken ile bağımlı değişken arasındaki ilişkiyi polinomal bir modelle daha iyi anlamayı sağlar.
