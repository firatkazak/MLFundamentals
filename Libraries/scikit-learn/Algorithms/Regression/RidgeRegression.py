import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# Veriler: Deneyim (yıl), Eğitim düzeyi (yıl) ve Maaş ($)
X = np.array([[1, 10], [2, 11], [3, 12], [4, 13], [5, 14], [6, 15]])  # Bağımsız değişkenler: Deneyim ve Eğitim düzeyi
y = np.array([30000, 35000, 40000, 45000, 50000, 60000])  # Bağımlı değişken: Maaş

# Ridge Regression model oluşturma
model = Ridge(alpha=1.0)  # Ridge modelini alpha=1.0 regularization parametresi ile oluşturuyoruz.
# Alpha parametresi Ridge Regression'da regularization gücünü belirler. Daha yüksek alpha, daha fazla regularization (model cezası) anlamına gelir.
# Ridge Regression, modeldeki aşırı öğrenmeyi (overfitting) azaltmak için regularization uygular.

model.fit(X, y)  # Modeli bağımsız değişkenler X ve bağımlı değişken y ile eğitiyoruz.
# fit() metodu: Model, verilen X ve y verileri üzerinden katsayıları öğrenir.
# X: Bağımsız değişkenler (deneyim ve eğitim verileri)
# y: Bağımlı değişken (maaş verileri)

# Tahmin yapma;
y_pred = model.predict(X)  # Eğitilen model ile maaş tahminleri yapıyoruz.
# predict() metodu: Model, verilen X verileri üzerinden tahmin edilen maaşları döndürür.

# 3D plot oluşturma;
fig = plt.figure()  # Bir figure (şekil) oluşturur, 3D grafiği eklemek için.
ax = fig.add_subplot(111, projection='3d')  # 3 boyutlu eksenler (axes) oluşturuyoruz.
# add_subplot(111): 1x1 griddeki 1. subplot anlamına gelir.
# projection='3d': 3 boyutlu grafiği belirtmek için kullanılır.

# Gerçek veri noktaları için dağılım grafiği (mavi)
ax.scatter(X[:, 0], X[:, 1], y, color='blue', label='Gerçek Maaşlar')  # Gerçek maaş verilerini dağılım grafiği (scatter plot) olarak çiziyoruz.
# X[:, 0]: X'in 0. sütunu (deneyim yılları)
# X[:, 1]: X'in 1. sütunu (eğitim yılları)
# y: Gerçek maaşlar
# color='blue': Noktaların rengi mavi
# label='Gerçek Maaşlar': Legend (açıklama) için etiket

# Tahmini değerler için yüzey grafiği (kırmızı)
x_surf, y_surf = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), num=10),
                             np.linspace(X[:, 1].min(), X[:, 1].max(), num=10))
# np.meshgrid: İki boyutlu bir grid (yüzey) oluşturur. Deneyim ve eğitim düzeyi arasında tahmini yüzeyi çizmek için.
# np.linspace: Deneyim ve eğitim verileri arasında 10 eşit aralıklı değer oluşturur (tahmin edilen yüzey için).

z_surf = model.coef_[0] * x_surf + model.coef_[1] * y_surf + model.intercept_  # Tahmin edilen z ekseni (maaş) değerlerini hesaplar.
# model.coef_[0]: Deneyim katsayısı (deneyimin maaşa etkisini temsil eder).
# model.coef_[1]: Eğitim katsayısı (eğitimin maaşa etkisini temsil eder).
# model.intercept_: Regresyon denkleminin sabit terimi (intercept).

ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.5)  # Tahmin edilen değerler için kırmızı bir yüzey grafiği oluşturur.
# plot_surface(): 3 boyutlu bir yüzey (surface) çizmek için kullanılır.
# color='red': Yüzeyin rengini kırmızı yapar.
# alpha=0.5: Yüzeyin yarı saydamlık seviyesini ayarlar (0 ile 1 arasında bir değer alır, burada %50 saydam).

# Eksen etiketleri ve grafik başlığı
ax.set_xlabel('Deneyim (yıl)')  # X ekseni etiketi
ax.set_ylabel('Eğitim (yıl)')  # Y ekseni etiketi
ax.set_zlabel('Maaş ($)')  # Z ekseni etiketi
ax.set_title('Ridge Regression')  # Grafiğe başlık ekler

plt.show()  # Grafiği ekrana getirir.

# Ridge Regression (L2 Regularization)
# Aşırı uyumlamayı (overfitting) önlemek için katsayılara bir ceza uygulayarak modeli düzenler.
# Çok değişkenli regresyon verilerini analiz etmede kullanılır. Amaç hata kareler toplamını minimize eden katsayıları, bu katsayılara bir ceza uygulayarak bulmaktır.
# Over-fittinge karşı dirençlidir. Çok boyutluluğa çözüm sunar.
# Tüm değişkenler ile model kurar, ilgisiz değişkenleri çıkarmaz sadece katsayılarını sıfıra yaklaştırır. Modeli kurarken alpha (ceza) için iyi bir değer bulmak gerekir.

# Overfitting/Underfitting Alegorisi:
# Overfitting: Bahçıvan kendi arazisine çok iyi çiçek eker fakat farklı bir bahçeye farklı bir çiçek ekmeye çalıştığında sapıtır.
# Underfitting: Bahçıvan çiçek ekmeyi yeterince öğrenemediği için çiçekleri düzgün ekemez, verim alamaz.

# Overfitting: Model, eğitim verisine aşırı uyum sağladığı için yeni verilere genelleme yapamaz.
# Bahçıvan her bir çiçeği bireysel olarak mükemmel bir şekilde yetiştirirken, genel bahçe yönetimini göz ardı eder.

# Underfitting: Model, verileri yeterince öğrenmediği için hem eğitim hem de test setlerinde zayıf performans gösterir.
# Bahçıvan, çiçeklerin temel ihtiyaçlarını göz ardı ederek başarılı olamaz.


# Bias/Variance Alegorisi:
# Bias: Okçular hedefe değil de sürekli hedefin soluna atış yapıyor. Sola doğru bir yanlılık var. Yüksek bias yüksek yanlılık demek.
# Variance: Okçular hedefe değil de sürekli sola, sağa, yukarıya ve aşağıya atış yapıyor. Yüksek varyans kötü atışı temsil ediyor.

# Bias (Yanlılık): Modelin hedefe olan sistematik sapmalarıdır. Yüksek bias, modelin verileri yeterince iyi öğrenemediğini ve genel olarak bir hata yaptığını gösterir.
# Okçuların tüm okları aynı yöne sapıyorsa, bu yüksek biası temsil eder.

# Variance (Varyans): Modelin hedefin etrafındaki dağınıklığıdır.
# Yüksek varyans, modelin veriye aşırı uyum sağladığını ve bu nedenle yeni verilere genelleme yapamadığını gösterir.
# Okçuların atışları çok dağınık ve düzensizse, bu yüksek varyansı temsil eder.

# Bias/Variance Dengesi:
# Mükemmel bir model, bias ve varyans arasında bir denge kurmalıdır.
# Yüksek bias, modelin genelleme yapamamasına neden olurken, yüksek varyans, modelin aşırı öğrenmesine neden olur.
# Amaç, bu iki hata türü arasındaki dengeyi bulmak ve hem biası hem de varyansı minimize ederek genel performansı artırmaktır.
