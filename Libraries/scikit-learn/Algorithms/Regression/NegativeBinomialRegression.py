# Gerekli kütüphaneleri içe aktarıyoruz
import numpy as np  # Sayısal hesaplamalar için
import matplotlib.pyplot as plt  # Görselleştirme için
import statsmodels.api as sm  # İstatistiksel modelleme için

# Veri oluşturma
np.random.seed(0)  # Rastgele sayıların tekrarlanabilir olması için sabitliyoruz
X = np.random.poisson(5, size=100)  # 100 örnek içeren Poisson dağılımından rastgele veri üretiyoruz
# Gerçek ilişki: y = X + hata (hata olarak Negative Binomial ekliyoruz)
y = np.random.negative_binomial(n=1, p=0.5, size=100) + X

# Tasarım matrisini oluşturma
X = sm.add_constant(X)  # X'e sabit terim (intercept) ekliyoruz

# Modeli tanımlama
neg_binom_model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=1.0))  # Negative Binomial regresyon modeli tanımlıyoruz

# Modeli eğitme
neg_binom_results = neg_binom_model.fit()  # Modeli verilerle eğitiyoruz

# Tahminler yapma
y_pred = neg_binom_results.predict(X)  # Negative Binomial modelimiz ile tahmin yapıyoruz

# Görselleştirme
plt.figure(figsize=(10, 6))  # Görsel boyutunu ayarlıyoruz
plt.scatter(X[:, 1], y, color='blue', label='Gerçek Veri')  # Gerçek verileri mavi renkle gösteriyoruz
plt.plot(X[:, 1], y_pred, color='green', label='Negative Binomial Regresyonu')  # Tahmin edilen Negative Binomial regresyon doğrusunu çiziyoruz
plt.xlabel('X')  # X eksenini etiketliyoruz
plt.ylabel('y')  # Y eksenini etiketliyoruz
plt.title('Negative Binomial Regression')  # Grafiğin başlığını belirliyoruz
plt.legend()  # Legend ekliyoruz
plt.grid(True)  # Izgara ekliyoruz
plt.show()  # Grafiği gösteriyoruz

# Negative Binomial Regresyon, sayma verilerinin modellemesinde kullanılan bir diğer önemli regresyon tekniğidir.
# Genellikle Poisson regresyonunun sınırlamalarını aşmak için tercih edilir.
# Bu model, özellikle olayların sayısının aşırı dağılım gösterdiği (overdispersion) durumlarda etkilidir.

# Temel Özellikleri:
# Aşırı Dağılım (Overdispersion): Poisson regresyonu, olayların sayısının ortalaması ile varyansının eşit olduğu varsayımına dayanır.
# Ancak gerçek hayatta bu durum sıkça gözlemlenmez. Negative Binomial Regresyon, varyansın ortalamadan daha büyük olduğu durumları modellemek için geliştirilmiştir.

# Dağılım: Negative Binomial dağılımı, belirli sayıda başarısızlığa ulaşmadan önceki başarı sayısını modellemek için kullanılır.
# Bu, belirli bir olayın gerçekleşme sayısını sayma durumunda etkilidir.

# İki Parametreli Yapı: Negative Binomial model, genellikle iki parametre içerir:
# Olayların ortalaması (μ)
# Olayların varyansı (σ²), bu varyans ortalamadan bağımsızdır ve olayların dağılımını daha esnek hale getirir.

# Logaritmik Link Fonksiyonu: Model, beklenen olay sayısının logaritması ile bağımsız değişkenler arasındaki ilişkiyi tanımlar.

# Kullanım Alanları:
# Epidemiyoloji: Hastalıkların sayısını modellemek için, çünkü genellikle aşırı varyansa sahip veriler gözlemlenir.
# Pazarlama: Müşteri davranışlarını incelemek için.
# Doğa Bilimleri: Ekosistemlerdeki tür çeşitliliği gibi sayma verilerinin analizinde.

# Avantajları:
# Aykırı değerler ve aşırı dağılım gösteren verilerle başa çıkmak için daha esnek bir model sunar.
# Poisson modeline kıyasla daha iyi uyum sağlar.
