import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

# Veriler: Deneyim (yıl), Eğitim düzeyi (yıl) ve Maaş ($)
X = np.array([[1, 10], [2, 11], [3, 12], [4, 13], [5, 14], [6, 15]])
# X, bağımsız değişkenleri temsil eder; her bir satır bir gözlemi, sütunlar ise değişkenleri (Deneyim ve Eğitim) ifade eder.

y = np.array([30000, 35000, 40000, 45000, 50000, 60000])
# y, bağımlı değişkeni (Maaş) temsil eder; her bir gözlem için maaş değerleri içerir.

# Lasso regresyon modelini oluşturma
model = Lasso(alpha=0.1)  # Lasso modelini oluşturuyoruz; alpha, modelin düzenleme gücünü kontrol eder.
# Küçük alpha değerleri daha az düzenleme yaparken, büyük değerler daha fazla düzenleme uygular.

model.fit(X, y)  # Modeli verilerle eğitiyoruz; X bağımsız değişkenler, y ise bağımlı değişkendir.

# Tahmin yapma
y_pred = model.predict(X)  # Modeli kullanarak X verileri için tahminler yapıyoruz ve sonuçları y_pred değişkenine atıyoruz.

# 3D plot
fig = plt.figure()  # Yeni bir figür oluşturuyoruz.
ax = fig.add_subplot(111, projection='3d')  # 3D alt grafik oluşturuyoruz; 111, 1x1'lik bir ızgarada 1. grafik anlamına gelir.

# Gerçek veri noktalarını çizme
ax.scatter(X[:, 0], X[:, 1], y, color='blue', label='Gerçek Maaşlar')
# Gerçek veri noktalarını mavi renkle 3D alanda çiziyoruz; X'in birinci ve ikinci sütununu (Deneyim ve Eğitim) kullanıyoruz.

# Tahmin edilen yüzey için ızgara oluşturma
x_surf, y_surf = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), num=10),
                             np.linspace(X[:, 1].min(), X[:, 1].max(), num=10))
# Meshgrid, tahmin edilen yüzey için ızgara noktaları oluşturur; Deneyim ve Eğitim için 10 eşit aralıklı nokta alıyoruz.

# Tahmin edilen yüzeyi hesaplama
z_surf = model.coef_[0] * x_surf + model.coef_[1] * y_surf + model.intercept_
# Tahmin edilen yüzeyin Z koordinatını hesaplıyoruz; model.coef_ regresyon katsayılarıdır, model.intercept_ ise sabittir.

# Tahmin edilen yüzeyi çizme
ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.5)
# Tahmin edilen yüzeyi kırmızı renkte ve %50 şeffaflıkla 3D alanda çiziyoruz.

# Etiketler ve Başlık
ax.set_xlabel('Deneyim (yıl)')  # X ekseni için etiket
ax.set_ylabel('Eğitim (yıl)')  # Y ekseni için etiket
ax.set_zlabel('Maaş ($)')  # Z ekseni için etiket
ax.set_title('Lasso Regression')  # Grafiğin başlığını belirtiyoruz.

plt.legend()  # Grafikteki etiketler için bir efsane (legend) oluşturuyoruz.
plt.show()  # Grafiği gösteriyoruz.

# Lasso Regression (L1 Regularization)
# Bazı özelliklerin katsayılarını tamamen sıfıra indirerek daha sade bir model oluşturur.

# Ürettiği modelin tahmin doğruluğunu ve yorumlanabilirliğini arttırmak için hem değişken seçimi hem de regularization yapar.
# Aynı ridge regresyonda olduğu gibi amaç hata kareler toplamını minimize eden katsayıları, katsayılara ceza uygularayarak bulmaktır.
# Fakat ridge regresyondan farklı olarak ilgisiz değişkenlerin katsayılarını sıfıra eşitler.
