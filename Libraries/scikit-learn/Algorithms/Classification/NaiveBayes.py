import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# Data: Experience (years) and whether they got promoted (1 = Yes, 0 = No)
X = np.array([[1], [2], [3], [4], [5], [6]])  # Bağımsız değişken: deneyim
y = np.array([0, 0, 0, 1, 1, 1])  # Bağımlı değişken: terfi (evet/hayır)

# Eğitim ve test setine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Naive Bayes modelini oluştur
model = GaussianNB()
model.fit(X_train, y_train)

# Tahmin yap
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)  # Olasılık tahminleri

print("Tahmin Edilen Terfiler: ", y_pred)
print("Tahmin Olasılıkları: ", y_prob)

# Görselleştirme için yeni X değerleri (daha fazla veri noktası ile)
X_new = np.linspace(start=0, stop=7, num=300).reshape(-1, 1)

# Tahmin edilen olasılıklar (sınıf 1 olma olasılığı)
y_prob_new = model.predict_proba(X_new)[:, 1]

# Verileri çizdir
plt.scatter(X, y, color='black', label='Gerçek Veriler')

# Naive Bayes eğrisini çizdir
plt.plot(X_new, y_prob_new, color='green', label='Naive Bayes Eğrisi')

# Grafiği ayarla
plt.xlabel('Deneyim (Yıl)')
plt.ylabel('Terfi Olasılığı')
plt.title('Naive Bayes: Deneyim ve Terfi Olasılığı')
plt.legend()
plt.show()

# Açıklamalar:
# GaussianNB(): Sürekli değişkenler için Naive Bayes modelini oluşturuyoruz. fit() metodu ile eğitim verisi üzerinde modelimizi eğitiyoruz.
# predict_proba(): Modelin yeni X değerleri için her bir sınıfın olasılık tahminlerini alıyoruz.
# plt.scatter(): Gerçek verileri (deneyim ve terfi olup olmaması) siyah noktalarla gösteriyoruz.
# plt.plot(): Naive Bayes modelinin tahmin ettiği olasılık eğrisini çizeriz.

# Görselleştirme:
# Naive Bayes Eğrisi: Eğri, modelin deneyim bazında terfi olasılıklarını nasıl tahmin ettiğini gösterir.
# Olasılıkların Dağılımı: Model, verinin dağılımına dayanarak kararlar alır; bu nedenle sonuçlar genellikle diğer modellerden farklı bir dağılım gösterebilir.
