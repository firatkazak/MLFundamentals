import numpy as np  # Sayısal hesaplamalar için
import matplotlib.pyplot as plt  # Görselleştirme için
import statsmodels.api as sm  # İstatistiksel modelleme için

# Veri oluşturma
np.random.seed(0)  # Rastgele sayıların tekrarlanabilir olması için sabitliyoruz
X = np.random.poisson(5, size=100)  # 100 örnek içeren Poisson dağılımından rastgele veri üretiyoruz
y = X + np.random.poisson(1, size=100)  # Gerçek ilişki: y = X + hata (hata olarak Poisson ekliyoruz)

# Tasarım matrisini oluşturma
X = sm.add_constant(X)  # X'e sabit terim (intercept) ekliyoruz

# Modeli tanımlama
poisson_model = sm.GLM(y, X, family=sm.families.Poisson())  # Poisson regresyon modeli tanımlıyoruz

# Modeli eğitme
poisson_results = poisson_model.fit()  # Modeli verilerle eğitiyoruz

# Tahminler yapma
y_pred = poisson_results.predict(X)  # Poisson modelimiz ile tahmin yapıyoruz

# Görselleştirme
plt.figure(figsize=(10, 6))  # Görsel boyutunu ayarlıyoruz
plt.scatter(X[:, 1], y, color='blue', label='Gerçek Veri')  # Gerçek verileri mavi renkle gösteriyoruz
plt.plot(X[:, 1], y_pred, color='green', label='Poisson Regresyonu')  # Tahmin edilen Poisson regresyon doğrusunu çiziyoruz
plt.xlabel('X')  # X eksenini etiketliyoruz
plt.ylabel('y')  # Y eksenini etiketliyoruz
plt.title('Poisson Regression')  # Grafiğin başlığını belirliyoruz
plt.legend()  # Legend ekliyoruz
plt.grid(True)  # Izgara ekliyoruz
plt.show()  # Grafiği gösteriyoruz

# Poisson Regresyon, sayısal verilere dayalı olarak sayma verilerini modellemek için kullanılan bir regresyon türüdür.
# Özellikle olayların belirli bir zaman diliminde veya alan içinde sayılması gerektiğinde kullanılır.
# Poisson dağılımına dayanan bu regresyon modeli, sayısal verilerin belirli bir ortalamaya göre dağıldığı durumlarda etkilidir.
#
# Temel Özellikleri:
# Olay Sayıları: Poisson Regresyon, genellikle belirli bir zaman diliminde veya belirli bir alanda meydana gelen olayların sayısını modellemek için kullanılır.
# Örneğin, bir hastaneye yapılan acil başvuruların sayısı veya bir web sitesine gelen ziyaretlerin sayısı.

# Dağılım: Model, yanıt değişkeninin Poisson dağılımına uyması gerektiğini varsayar. Yani, olayların ortalama sayısı, verilerin dağılımını belirler.

# Logaritmik Link Fonksiyonu: Poisson Regresyon, olay sayısının beklenen değerinin logaritmasını kullanarak model kurar.
