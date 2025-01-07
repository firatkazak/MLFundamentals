import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Rastgele veri oluşturma
np.random.seed(42)  # Sonuçların tekrarlanabilir olması için
X = np.random.exponential(scale=2, size=100)  # Bağımsız değişken
y = np.random.gamma(shape=2, scale=1, size=100) + X  # Bağımlı değişken

# Veri çerçevesi oluşturma
data = pd.DataFrame({'X': X, 'y': y})

# Gamma regresyon modeli, log bağlantı fonksiyonu ile
model = sm.GLM(data['y'], sm.add_constant(data['X']), family=sm.families.Gamma(link=sm.families.links.Log())).fit()

# Model tahminleri
data['predicted'] = model.predict(sm.add_constant(data['X']))

# Sonuçları görselleştirme
plt.figure(figsize=(10, 6))
plt.scatter(data['X'], data['y'], color='blue', label='Gerçek Değerler', alpha=0.6)
plt.plot(data['X'], data['predicted'], color='red', label='Tahmin Edilen Değerler', linewidth=2)
plt.title('Gamma Regresyon Modeli')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()

# Gamma regresyon,
# Sürekli ve pozitif değerler için uygun bir regresyon modelidir.
# Özellikle, bağımlı değişkenin pozitif ve genellikle sağa çarpık dağılımlara sahip olduğu durumlarda kullanılır.

# ANA ÖZELLİKLERİ;

# Dağılım: Gamma dağılımı, pozitif sürekli bir dağılımdır ve genellikle süre, miktar veya yoğunluk gibi pozitif verilere uygulanır.
# Özellikle sağa çarpık verilerde iyi performans gösterir.

# Model: Gamma regresyonu, bağımlı değişkenin Gamma dağılımına uyum sağlayacak şekilde bağımsız değişkenlerle ilişkilendirilmesini amaçlar.
# Model, bağımlı değişkenin (genellikle y) bağımsız değişkenler (genellikle X) cinsinden açıklanmasını sağlar.

# Link Fonksiyonu: Gamma regresyonu, bağımlı değişkenin beklenen değerinin bağımsız değişkenler ile bir ilişki kurmak için bir bağlantı fonksiyonu kullanır.
# En yaygın olarak kullanılan bağlantı fonksiyonu log fonksiyonudur. Bu, tahminlerin pozitif olmasını garanti eder.

# Gamma Regresyonun Kullanım Alanları
# Ekonomi: Maliyet tahminleri veya gelir gibi pozitif verilerin modellenmesi.
# Biyolojik Veriler: Büyüme oranları, hastalık süreleri veya sağlık verileri gibi pozitif dağılımlara sahip biyolojik verilerin analizi.
# Mühendislik: Arıza süreleri, yaşam süreleri veya yoğunluk hesaplamaları gibi mühendislik uygulamaları.

# Modelin Uygulanması: Bir Gamma regresyon modeli oluşturmak için genellikle şu adımlar izlenir;
# Veri Hazırlığı: Bağımlı ve bağımsız değişkenlerinizi belirleyin. Bağımlı değişkenin yalnızca pozitif değerlere sahip olduğundan emin olun.
# Modeli Tanımlama: Gamma regresyonunu tanımlayın ve uygun bağlantı fonksiyonunu seçin.
# Modelin Eğitilmesi: Verileri kullanarak modeli eğitin.
# Tahminler ve Değerlendirme: Modelden tahminler alın ve modelin başarımını değerlendirin.

# Örnek;
# Eğer bir şirketin satışlarının, reklam harcamalarına bağlı olarak nasıl değiştiğini incelemek istiyorsanız, satış miktarlarını bağımlı değişken olarak alabilir
# ve reklam harcamalarını bağımsız değişken olarak kullanabilirsiniz. Gamma regresyonu, bu pozitif değerleri etkili bir şekilde modellemenizi sağlar.

# Sonuç
# Gamma regresyon, pozitif veriler üzerinde çalışırken etkili bir araçtır ve birçok alanda uygulanabilir.
# Bu model, verilerinizin doğasına uygun olduğunda, iyi sonuçlar verebilir.
