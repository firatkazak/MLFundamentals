# Gerekli kütüphaneleri içe aktarıyoruz
import numpy as np  # Sayısal hesaplamalar için
import matplotlib.pyplot as plt  # Görselleştirme için
from sklearn.linear_model import RANSACRegressor, LinearRegression  # RANSAC ve Basit Doğrusal Regresyon için

# Veri oluşturma
np.random.seed(0)
X = np.random.normal(size=(100, 1))  # 100 rastgele normal dağılımlı örnek
y = 2 * X.squeeze() + 1 + np.random.normal(size=100)  # Gerçek ilişki: y = 2x + 1 + hata

# Aşırı değer ekleme
y[::10] += 10  # Her 10. değere 10 ekleyerek aşırı değerler oluşturuyoruz

# Modeli tanımlama
linear_reg = LinearRegression()  # Basit doğrusal regresyon modeli
ransac = RANSACRegressor()  # RANSAC algoritmasını kullanarak robust regresyon modeli oluşturuyoruz

# Modeli eğitme
ransac.fit(X, y)  # Modeli X ve y verileri ile eğitiyoruz

# Tahminler yapma
y_ransac = ransac.predict(X)  # RANSAC modelimiz ile tahmin yapıyoruz

# Aşırı değerleri belirleme
inlier_mask = ransac.inlier_mask_  # Modelin içeriğine göre aşırı değer maskesi oluşturuyoruz
outlier_mask = np.logical_not(inlier_mask)  # Aşırı değer maskesini alıyoruz

# Görselleştirme
plt.figure(figsize=(10, 6))  # Görsel boyutunu ayarlıyoruz
plt.scatter(X[inlier_mask], y[inlier_mask], color='blue', label='Inliers')  # İç noktaları mavi renkle gösteriyoruz
plt.scatter(X[outlier_mask], y[outlier_mask], color='red', label='Outliers')  # Aşırı değerleri kırmızı renkle gösteriyoruz
plt.plot(X, y_ransac, color='green', label='RANSAC Regression')  # RANSAC regresyon doğrusunu çiziyoruz
plt.legend()  # Legend ekliyoruz
plt.xlabel('X')  # X eksenini etiketliyoruz
plt.ylabel('y')  # Y eksenini etiketliyoruz
plt.title('Robust Regression with RANSAC')  # Grafiğin başlığını belirliyoruz
plt.grid(True)  # Izgara ekliyoruz
plt.show()  # Grafiği gösteriyoruz

# Robust, veri kümesindeki aykırı değerlerin (outliers) regresyon analizi üzerindeki olumsuz etkilerini azaltmak amacıyla geliştirilmiş bir regresyon tekniğidir.
# Geleneksel regresyon yöntemleri, özellikle en küçük kareler yöntemine dayalı olanlar, aykırı değerlerden ciddi şekilde etkilenebilir.
# Robust Regression ise bu durumu göz önünde bulundurarak tasarlanmıştır.

# Temel Özellikleri:
# Aykırı Değerlere Dayanıklılık: Robust Regression, veri kümesindeki aykırı değerlerden daha az etkilenerek daha güvenilir sonuçlar sağlar.

# Farklı Kayıp Fonksiyonları: Geleneksel regresyonda hata karelerinin toplamı minimize edilirken,
# robust regresyonda genellikle farklı kayıp fonksiyonları (örneğin, mutlak hata) kullanılır.

# Örnek Yöntemleri: En yaygın robust regresyon yöntemleri arasında;

# M-estimators: Aykırı değerlere karşı dayanıklı estimatörler.
# Least Absolute Deviations (LAD): Mutlak hata toplamını minimize eden bir yaklaşım.
# RANSAC (Random Sample Consensus): Aykırı değerleri otomatik olarak tanımlayarak sadece "iyi" verilerle model oluşturma.

# Kullanım Alanları:
# Finansal Analiz: Aykırı değerlerin sıkça bulunduğu finansal verilerde.
# Hastalık Araştırmaları: Sağlık verilerinde, hastaların verileri bazen aşırı değerler içerebilir.
# Mühendislik Uygulamaları: Veri toplama hatalarının bulunduğu mühendislik çalışmalarında.

# RANSAC (Random Sample Consensus):
# RANSAC, veri setindeki iç noktaları (inliers) belirleyerek bu noktalar üzerinden regresyon modeli oluşturur ve aşırı değerleri (outliers) göz ardı eder.
