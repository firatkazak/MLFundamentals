import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Iris veri setini yükle
data = load_iris()
X = data.data[:, :2]  # İlk iki özelliği al
y = data.target

# Yalnızca iki sınıfı (0 ve 1) kullan
mask = y != 2  # Sadece setosa (0) ve versicolor (1) sınıflarını al
X = X[mask]
y = y[mask]

# Eğitim ve test setine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# SVM modelini oluştur
svm_model = SVC(kernel='linear', random_state=0)
svm_model.fit(X_train, y_train)

# Tahmin yap
y_pred = svm_model.predict(X_test)

# Sonuçları yazdır
print("Doğruluk Skoru:", accuracy_score(y_test, y_pred))

# Sonuçları görselleştir
plt.figure(figsize=(10, 6))
colors = ['red', 'green']
markers = ['o', 'x']

# Eğitim verilerini ve test verilerini görselleştir
for i, color in zip(np.unique(y), colors):
    plt.scatter(X_train[y_train == i, 0], X_train[y_train == i, 1],
                color=color, alpha=0.5, label=f'Eğitim Sınıf {i}', marker=markers[i])
    plt.scatter(X_test[y_test == i, 0], X_test[y_test == i, 1],
                color=color, s=100, label=f'Test Sınıf {i}', marker='o', edgecolor=color)

# SVM karar sınırını çiz
xlim = plt.xlim()
ylim = plt.ylim()

xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100),
                     np.linspace(ylim[0], ylim[1], 100))

# Karar fonksiyonunu hesapla
Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)  # Z'yi yeniden şekillendir

# Karar sınırını çiz
plt.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.5, linewidths=2)

plt.title('Support Vector Machines Sonuçları')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.legend()
plt.grid()
plt.show()
