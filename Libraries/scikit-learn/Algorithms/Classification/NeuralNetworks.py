import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.model_selection import train_test_split

# Data: Experience (years) and whether they got promoted (1 = Yes, 0 = No)
X = np.array([[1], [2], [3], [4], [5], [6]])  # Bağımsız değişken: deneyim
y = np.array([0, 0, 0, 1, 1, 1])  # Bağımlı değişken: terfi (evet/hayır)

# Eğitim ve test setine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Yapay Sinir Ağı modelini oluştur
model = Sequential()
model.add(Input(shape=(1,)))  # Giriş katmanı
model.add(Dense(5, activation='relu'))  # Gizli katman
model.add(Dense(1, activation='sigmoid'))  # Çıkış katmanı

# Modeli derle
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modeli eğit
model.fit(X_train, y_train, epochs=100, verbose=0)

# Tahmin yap
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)  # 0.5 eşiğini kullanarak sınıf tahmini
y_prob = model.predict(X_test)  # Olasılık tahminleri

print("Tahmin Edilen Terfiler: ", y_pred_classes.flatten())
print("Tahmin Olasılıkları: ", y_prob.flatten())

# Görselleştirme için yeni X değerleri (daha fazla veri noktası ile)
X_new = np.linspace(0, 7, 300).reshape(-1, 1)

# Tahmin edilen olasılıklar (sınıf 1 olma olasılığı)
y_prob_new = model.predict(X_new)

# Verileri çizdir
plt.scatter(X, y, color='black', label='Gerçek Veriler')

# Yapay Sinir Ağı eğrisini çizdir
plt.plot(X_new, y_prob_new, color='red', label='Yapay Sinir Ağı Eğrisi')

# Grafiği ayarla
plt.xlabel('Deneyim (Yıl)')
plt.ylabel('Terfi Olasılığı')
plt.title('Yapay Sinir Ağı: Deneyim ve Terfi Olasılığı')
plt.legend()
plt.show()

# Açıklamalar:

# Model Tanımı:
# Sequential(): Modelin sıralı bir yapı olduğunu belirtir.
# Dense(): Tam bağlı (fully connected) katmanları ekler.
# İlk katman 5 nöron ve relu aktivasyon fonksiyonu kullanırken, çıkış katmanı 1 nöron ve sigmoid aktivasyon fonksiyonu kullanır.

# Model Derleme:
# binary_crossentropy: İki sınıflı problemler için kayıp fonksiyonu.
# adam: Model optimizasyonu için kullanılan popüler bir optimizasyon algoritmasıdır.

# Modeli Eğitme:
# fit(): Modeli eğitim verileri ile eğitir. epochs=100 ifadesi, modelin 100 defa tüm veri seti üzerinden geçeceği anlamına gelir.

# Tahmin Yapma:
# Modelden tahmin alır ve olasılıkları sınıf tahminlerine dönüştürmek için 0.5 eşiğini kullanır.

# Görselleştirme:
# plt.plot(): Yapay Sinir Ağı tarafından tahmin edilen olasılık eğrisini çizer.

# Sonuç:
# Bu örnek, basit bir yapay sinir ağı kullanarak deneyime dayalı terfi olasılıklarını tahmin etme yeteneğini gösterir.
# Görselleştirme, modelin tahmin ettiği olasılıkların veriler üzerindeki etkisini açık bir şekilde ortaya koyar.
