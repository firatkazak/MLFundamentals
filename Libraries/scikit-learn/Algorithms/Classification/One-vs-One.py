import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import combinations

# Iris veri setini yükle
data = load_iris()
X = data.data
y = data.target

# Eğitim ve test setine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# One-vs-One modeli için modelin listesini tut
models = []
classes = np.unique(y)
n_classes = len(classes)

# Sınıf çiftlerini oluştur
class_pairs = list(combinations(classes, r=2))

# Her sınıf çifti için ayrı bir model eğit
for class1, class2 in class_pairs:
    # İlgili sınıf çiftine ait veriyi seç
    indices = np.where((y_train == class1) | (y_train == class2))
    X_train_pair = X_train[indices]
    y_train_pair = y_train[indices]

    # Sınıf etiketlerini 0 ve 1 olarak yeniden düzenle
    y_train_pair = np.where(y_train_pair == class1, 0, 1)

    # Modeli oluştur
    model = Sequential()
    model.add(Input(shape=(X_train_pair.shape[1],)))  # Giriş katmanı
    model.add(Dense(5, activation='relu'))  # Gizli katman
    model.add(Dense(1, activation='sigmoid'))  # Çıkış katmanı

    # Modeli derle
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Modeli eğit
    model.fit(X_train_pair, y_train_pair, epochs=100, verbose=0)  # Her model yalnızca iki sınıfı öğrenir
    models.append((model, class1, class2))

# Tahmin yap
y_pred = np.zeros((X_test.shape[0], n_classes))

for model, class1, class2 in models:
    pred = model.predict(X_test)
    # Tahminleri sınıflara yerleştir
    y_pred[:, class1] += (pred.flatten() < 0.5)  # class1 için 0, class2 için 1
    y_pred[:, class2] += (pred.flatten() >= 0.5)  # class2 için 1

# En yüksek oy alan sınıfı seç
y_pred_classes = np.argmax(y_pred, axis=1)

# Doğruluk hesapla
accuracy = accuracy_score(y_test, y_pred_classes)
print("Doğruluk Skoru:", accuracy)

# Sonuçları görselleştir
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_classes, cmap='viridis', label='Tahmin Edilen Sınıflar')
plt.title('One-vs-One Sınıflandırma Sonuçları')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.colorbar(label='Sınıf')
plt.show()

# Alegori:
# Burada öğretmen, iki öğrenciyi bir araya getiriyor ve onlara sadece bir soru soruyor.
# Örneğin, "Ahmet ve Ayşe, futbol mu yoksa basketbol mu daha iyi?" diye soruyor.
# Bu işlemi tüm öğrenci çiftleri için yapıyor. Her bir çiftte kazanan belirleniyor ve sonunda hangi sporun daha popüler olduğunu belirliyor.

# One-vs-One (OvO): Her sınıf çiftini karşılaştırmak için bir model eğitilir.
# Yukarıdaki örnekte, A vs B, A vs C, B vs C gibi üç ayrı model vardır.
# Her bir model, hangi sınıfın diğerine göre daha iyi olduğunu belirler.
