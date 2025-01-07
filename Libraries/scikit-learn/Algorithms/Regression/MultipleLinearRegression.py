import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Veriler: Deneyim (yıl), Eğitim düzeyi (yıl), Maaş ($)
X = np.array([[1, 10], [2, 11], [3, 12], [4, 13], [5, 14], [6, 15]])  # Bağımsız değişkenler: deneyim (X[:, 0]) ve eğitim düzeyi (X[:, 1])
y = np.array([30000, 35000, 40000, 45000, 50000, 60000])  # Bağımlı değişken: maaş

# Model oluşturma
model = LinearRegression()  # LinearRegression sınıfından bir model nesnesi oluşturuyoruz.
# LinearRegression()'daki bazı önemli parametreler:
# fit_intercept: True olarak bırakılırsa modelin sabit (intercept) terimini öğrenir.
# normalize: Eğer True olursa model verileri fit işlemi sırasında normalize eder. (Varsayılan False)

model.fit(X, y)  # Modeli bağımsız değişkenler X ve bağımlı değişken y ile eğitiyoruz.
# fit() metodu:
# X: Bağımsız değişkenlerin dizisi (input), yani deneyim ve eğitim verileri.
# y: Bağımlı değişken (output), yani maaş verileri.

# Tahmin yapma
y_pred = model.predict(X)  # Eğitilen model ile maaş tahminleri yapıyoruz.
# predict() metodu:
# X: Modelin kullanacağı bağımsız değişkenler (deneyim ve eğitim verileri).

# 3D Grafik oluşturma
fig = plt.figure()  # Bir figure (şekil) oluşturur, üzerine 3D grafik ekleyebilmek için.
ax = fig.add_subplot(111, projection='3d')  # 3 boyutlu eksenler (axes) oluşturur.
# add_subplot(111): 1x1 gridin 1. subplot'u anlamına gelir, burada tek bir eksen kullanıyoruz.
# projection='3d': 3D grafik oluşturmak için gereken parametre.

# Gerçek veri noktaları (mavi noktalar)
ax.scatter(X[:, 0], X[:, 1], y, color='blue', label='Gerçek Maaşlar')  # Gerçek maaş verilerini 3D scatter plot (dağılım grafiği) olarak çizer.
# X[:, 0]: X'in 0. sütunu, yani deneyim yılları.
# X[:, 1]: X'in 1. sütunu, yani eğitim yılları.
# y: Gerçek maaş değerleri.
# color='blue': Noktaların rengini mavi yapar.
# label='Gerçek Maaşlar': Grafik açıklaması (legend) için etiket.

# Tahmin edilen düzlem (kırmızı çizgi)
x_surf, y_surf = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 10),  # Deneyim verileri için yüzey (grid) oluşturur.
                             np.linspace(X[:, 1].min(), X[:, 1].max(), 10))  # Eğitim verileri için yüzey (grid) oluşturur.
# np.meshgrid: İki bağımsız değişkenin üzerinde tahmin edilen bir yüzey (surface) oluşturmak için kullanılır.
# np.linspace(): Deneyim ve eğitim değerleri arasında eşit aralıklarla 10 değer üretir.

z_surf = model.coef_[0] * x_surf + model.coef_[1] * y_surf + model.intercept_  # Doğrusal regresyon düzlemi (surface) için z değerlerini hesaplar.
# model.coef_[0]: Deneyim katsayısı.
# model.coef_[1]: Eğitim katsayısı.
# model.intercept_: Doğrusal regresyon denklemindeki sabit terim.

ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.5)  # 3D yüzeyi (surface) çizer.
# plot_surface(): 3 boyutlu bir yüzey (surface) oluşturur.
# color='red': Yüzeyi kırmızı renkte çizer.
# alpha=0.5: Yüzeyin yarı saydamlık seviyesini ayarlar.

# Eksen etiketleri
ax.set_xlabel('Deneyim (yıl)')  # X ekseninin etiketini ayarlar.
ax.set_ylabel('Eğitim (yıl)')  # Y ekseninin etiketini ayarlar.
ax.set_zlabel('Maaş ($)')  # Z ekseninin etiketini ayarlar.
ax.set_title('Multiple Linear Regression')  # Grafiğe başlık ekler.

plt.show()  # Grafiği ekrana getirir.

# 6 yıl deneyim ve 10 yıl eğitim düzeyi olan kişi için maaş tahmini
new_data = np.array([[6, 10]])  # Yeni veriyi 2D array olarak tanımlıyoruz
predicted_salary = model.predict(new_data)  # Model ile tahmin yapıyoruz

print(f"6 yıl deneyimi ve 10 yıl eğitim düzeyi olan bir kişinin tahmin edilen maaşı: {predicted_salary[0]:.2f} $")
# 6 yıl deneyimi ve 10 yıl eğitim düzeyi olan bir kişinin tahmin edilen maaşı: 43333.33 $

# Multiple Linear Regression
# Birden fazla bağımsız değişkeni aynı anda kullanarak bağımlı değişkeni tahmin etmeye çalışır.
