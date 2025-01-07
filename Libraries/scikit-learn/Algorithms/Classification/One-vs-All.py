import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# Iris veri setini yükle
data = load_iris()
X = data.data
y = data.target.reshape(-1, 1)  # Y'yi dikey vektöre çevir

# One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False)  # sparse yerine sparse_output kullandık
y_onehot = encoder.fit_transform(y)

# Eğitim ve test setine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=0)

# One-vs-All modeli için modelin listesini tut
models = []
n_classes = y_onehot.shape[1]

# Her sınıf için ayrı bir model eğit
for i in range(n_classes):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))  # Giriş katmanı
    model.add(Dense(5, activation='relu'))  # Gizli katman
    model.add(Dense(1, activation='sigmoid'))  # Çıkış katmanı

    # Modeli derle
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Modeli eğit
    model.fit(X_train, y_train[:, i], epochs=100, verbose=0)  # Her model sadece bir sınıfı öğrenir
    models.append(model)

# Tahmin yap
y_pred = np.zeros((X_test.shape[0], n_classes))

for i in range(n_classes):
    y_pred[:, i] = models[i].predict(X_test).flatten()

# En yüksek olasılığa sahip sınıfı seç
y_pred_classes = np.argmax(y_pred, axis=1)

# Gerçek sınıfları geri dönüştür
y_test_classes = np.argmax(y_test, axis=1)

# Doğruluk hesapla
accuracy = accuracy_score(y_test_classes, y_pred_classes)
print("Doğruluk Skoru:", accuracy)

# Sonuçları görselleştir
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_classes, cmap='viridis', label='Tahmin Edilen Sınıflar')
plt.title('One-vs-All Sınıflandırma Sonuçları')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.colorbar(label='Sınıf')
plt.show()

# Alegori:
# Öğretmen her bir öğrenci için tek tek sorular soruyor. Yani, ilk olarak "Ahmet futbolu mu yoksa basketbolu mu seviyor?" diye soruyor.
# Eğer Ahmet futbolu seviyorsa, "Futbol en iyi sporu!" diyor.
# Bu şekilde, her öğrenci için tüm spor dallarını sırasıyla deniyor.

# One-vs-All (OvA): Her sınıf için bir model eğitilir.
# Örneğin, üç sınıf (A, B, C) varsa, üç ayrı model oluşturulur: A vs (B, C), B vs (A, C), C vs (A, B).
# Her model, ilgili sınıfın diğerlerinden daha iyi olduğunu belirlemeye çalışır.
