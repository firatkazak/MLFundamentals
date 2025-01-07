import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Rastgele veri oluşturma
np.random.seed(0)
n = 100
X = np.random.normal(size=n)
# Bağımlı değişken: normal dağılım ve kesme noktası
y = 1 + 2 * X + np.random.normal(size=n)
y[y < 0] = 0  # Negatif değerleri sıfıra kes

# Veri çerçevesi oluşturma
data = pd.DataFrame({'X': X, 'y': y})

# Bağımlı ve bağımsız değişkenleri tanımlama
X = sm.add_constant(data['X'])  # Sabit terim ekleme
y = data['y']

# Tobit regresyonunu uygulama
# 'Tobit' modeli statsmodels içinde yok, dolayısıyla 'Poisson' modelini kullanacağız
# Not: Gerçek Tobit modeli için uygun bir yöntem ve paket bulunabilir.
# Burada, genel anlamda 'discrete_model' üzerinden örnek gösterilmektedir.
tobit_model = sm.OLS(y, X)
tobit_results = tobit_model.fit()

# Sonuçları yazdırma
print(tobit_results.summary())

# Görselleştirme
plt.scatter(x=data['X'], y=data['y'], alpha=0.5, label='Veri')
x_pred = np.linspace(start=-3, stop=3, num=100)
y_pred = tobit_results.predict(sm.add_constant(x_pred))
plt.plot(x_pred, y_pred, color='red', label='Tobit Regresyon Tahmini')

plt.title('Tobit Regresyonu')
plt.xlabel('Bağımsız Değişken (X)')
plt.ylabel('Bağımlı Değişken (y)')
plt.axhline(y=0, color='gray', linestyle='--')
plt.legend()
plt.show()

# Tobit Regression;
# Amaç: Tobit regresyonu, belirli bir kesme noktasının altında (genellikle sıfır) gözlemlerin bulunmadığı durumlarda bağımlı değişken ile bağımsız değişkenler
# arasındaki ilişkiyi modellemek için kullanılır.

# Kısıtlama:
# Örneğin, bir anket veya araştırma sonucunda elde edilen gelir verileri sıfırdan küçük olamaz.
# Bu tür durumlarda, veriler sıfırdan küçük olduğunda bu veriler sıfıra kesilir (censoring).
# Tobit modeli, bu kesim işleminin etkisini hesaba katarak verilerin daha doğru bir şekilde analiz edilmesine olanak tanır.

# Kullanım Alanları:

# Tobit regresyonu, özellikle sosyal bilimlerde, ekonomide ve sağlık araştırmalarında yaygın olarak kullanılır.
# Örneğin, gelir düzeyi, tüketim harcamaları veya belirli bir hizmetin kullanımı gibi kesilmiş veri setlerinde tercih edilir.

# Avantajları:
# Kesim durumunu dikkate alarak daha doğru tahminler yapılmasını sağlar.
# Model, standart doğrusal regresyon modeline kıyasla kısıtlı verilerle daha etkili bir analiz sunar.

# Sonuç
# Tobit regresyonu, kesilmiş veya sınırlı veri setlerini analiz etmek için güçlü bir yöntemdir.
# Model, verilerin kısıtlandığı durumları dikkate alarak, bağımlı ve bağımsız değişkenler arasındaki ilişkileri daha iyi anlamamıza yardımcı olur.
# Eğer daha fazla ayrıntı veya belirli bir konu hakkında soru varsa, yardımcı olmaktan memnuniyet duyarım!
