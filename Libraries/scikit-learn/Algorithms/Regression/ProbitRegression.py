import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Rastgele veri oluşturma
np.random.seed(0)
n = 100
X = np.random.normal(size=n)

# Bağımlı değişken: normal dağılım ve kesme noktası
# Gerçek olayların olasılığı (probit fonksiyonu ile belirlenmiş)
y_prob = 1 / (1 + np.exp(-X))  # Sigmoid fonksiyonu
y = np.random.binomial(n=1, p=y_prob)  # İkili sonuç

# Veri çerçevesi oluşturma
data = pd.DataFrame({'X': X, 'y': y})

# Bağımlı ve bağımsız değişkenleri tanımlama
X = sm.add_constant(data['X'])  # Sabit terim ekleme
y = data['y']

# Probit regresyonunu uygulama
probit_model = sm.Probit(y, X)
probit_results = probit_model.fit()

# Sonuçları yazdırma
print(probit_results.summary())

# Görselleştirme
plt.scatter(data['X'], data['y'], alpha=0.5, label='Veri')
x_pred = np.linspace(start=-3, stop=3, num=100)
y_pred_prob = probit_results.predict(sm.add_constant(x_pred))
plt.plot(x_pred, y_pred_prob, color='red', label='Probit Regresyon Tahmini')

plt.title('Probit Regresyonu')
plt.xlabel('Bağımsız Değişken (X)')
plt.ylabel('Bağımlı Değişken (y)')
plt.axhline(y=0.5, color='gray', linestyle='--')  # Eşik değeri
plt.legend()
plt.show()

# Probit Regresyonu;

# Amaç:
# Bağımlı değişkenin iki kategoriye ayrıldığı durumlarda (evet/hayır gibi) bağımsız değişkenler ile bağımlı değişken arasındaki ilişkiyi modellemek için kullanılır.

# Modelin Temeli:
# Probit regresyonu, ikili bir sonuç elde etmek için normal dağılım fonksiyonunu kullanır.
# Modelin temel mantığı, bağımsız değişkenlerin belirli bir olayın (örneğin, bir ürün satın alma) gerçekleşme olasılığını tahmin etmektir.

# Kullanım Alanları:
#
# Probit regresyonu, sağlık araştırmaları, sosyal bilimler, pazarlama ve ekonomi gibi birçok alanda kullanılır.
# Örn, bir hastanın bir tedaviye yanıt verip vermeyeceğini tahmin etmek veya bir müşteri segmentinin bir ürün satın alma olasılığını değerlendirmek için kullanılabilir.

# Avantajları:
# Probit modeli, bağımlı değişkenin ikili olduğu durumlarda, bağımsız değişkenlerin etkisini daha iyi analiz etmeye olanak tanır.
# Model, bağımlı değişkenin her iki kategorisi için de normal dağılım varsayımını dikkate alır, bu da bazı durumlarda daha doğru tahminler sağlar.

# Sonuç;
# Probit regresyonu, ikili sonuçlar için etkili bir modelleme yöntemidir ve bağımsız değişkenlerin bu sonuçlar üzerindeki etkisini anlamak için kullanılır.
# Model, sosyal bilimlerden ekonomiye kadar geniş bir yelpazede uygulama alanına sahiptir.
# Daha fazla bilgi veya belirli bir konu hakkında sorularınız varsa, yardımcı olmaktan memnuniyet duyarım!
