import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Rastgele veri oluşturma
np.random.seed(42)  # Rastgelelik için tohum ayarı
X = np.random.rand(100, 1) * 10  # Bağımsız değişken: 100 rastgele değer (0-10 arası)
y = 2 * X.squeeze() + np.random.normal(0, 3, 100)  # Hedef değişken: X'in 2 katı + normal dağılım hatası

# DataFrame oluşturma
df = pd.DataFrame({'X': X.squeeze(), 'y': y})  # Bağımsız ve bağımlı değişkenleri içeren DataFrame

# Quantile Regression modeli için kuantili belirleme (0.5, yani medyan)
quantile = 0.5  # İlgili kuantilin değeri

# Kuantile regresyonu uygulama
model = sm.QuantReg(df['y'], sm.add_constant(df['X']))  # Model tanımlama
result = model.fit(q=quantile)  # Modeli kuantile ile fit etme

# Tahminleri elde etme
y_pred = result.predict(sm.add_constant(df['X']))  # Tahmin edilen y değerleri

# Sonuçları görselleştirme
plt.scatter(df['X'], df['y'], color='blue', label='Gerçek Değerler')  # Gerçek veri noktaları
plt.plot(df['X'], y_pred, color='red', label=f'Quantile Regression (q={quantile})')  # Regresyon sonucu
plt.xlabel('X Değeri')  # X ekseni etiketi
plt.ylabel('Y Değeri')  # Y ekseni etiketi
plt.title('Quantile Regression')  # Başlık
plt.legend()  # Legend ekleme
plt.show()  # Grafiği gösterme

# Sonuçları yazdırma
print(result.summary())  # Model sonuçlarının özeti

# Quantile Regression

# Quantile Regression: Bağımlı değişkenin belirli kuantillerinin (örneğin, medyan, alt %25 veya üst %75) tahmin edilmesine olanak tanıyan bir regresyon tekniğidir.
# Bu yöntem, özellikle şu durumlarda faydalıdır:

# 1. Veri Dağılımını Anlamak;
# Klasik regresyon yöntemleri ortalama değerleri tahmin ederken, Quantile Regression, bağımlı değişkenin farklı bölgelerinde nasıl davrandığını anlamaya yardımcı olur.
# Bu sayede, verinin dağılımı hakkında daha ayrıntılı bilgi elde edilebilir.

# 2. Aşırı Değerlerden Etkilenmeme;
# Klasik regresyon, aşırı değerlerden (outlier) etkilenebilir. Kuantile regresyon, belirli kuantilleri tahmin ettiği için bu tür değerlerin etkisi daha azdır.
# Örneğin, medyan regresyon, ortalamadan daha az etkilenir.

# 3. Koşullu Dağılımı Analiz Etme;
# Kuantile regresyonı, bağımsız değişkenlerin bağımlı değişken üzerindeki etkisinin, farklı kuantillerde nasıl değiştiğini gösterir.
# Bu, modellemenin daha esnek ve detaylı olmasını sağlar.

# 4. Politika ve Ekonomi Analizleri;
# Kuantile regresyon, özellikle sosyal bilimlerde ve ekonomide, gelir dağılımı gibi konularda kullanışlıdır.
# Örneğin, belirli bir eğitim seviyesinin, gelir düzeyinin en düşük %25’i veya en yüksek %75’i üzerindeki etkisini incelemek için yararlıdır.

# 5. Çeşitli Regresyon Modelleri;
# Kuantile regresyon, birden fazla regresyon modelinin aynı anda değerlendirilmesine olanak tanır.
# Bu sayede, modelin farklı kuantillerdeki davranışlarını karşılaştırabilirsiniz.

# Özet;
# Kuantile regresyon, verilerin yalnızca ortalama değil, aynı zamanda farklı dağılımlarını ve kuantillerini modelleme imkanı sunarak daha kapsamlı bir analiz sağlar.
# Bu özellikleri sayesinde, özellikle karmaşık veri setlerinde ve aşırı değerlerin olduğu durumlarda güçlü bir araçtır.
