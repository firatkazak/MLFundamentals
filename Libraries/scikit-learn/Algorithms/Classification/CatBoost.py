import matplotlib.pyplot as plt
import numpy as np
import shutil
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

# Iris veri setini yükle
data = load_iris()
X = data.data
y = data.target

# Eğitim ve test setine ayır (sadece ilk iki özelliği kullanarak görselleştirme yapalım)
X_train, X_test, y_train, y_test = train_test_split(X[:, :2], y, test_size=0.2, random_state=0)

# CatBoost modelini oluştur
cat_model = CatBoostClassifier(silent=True)
cat_model.fit(X_train, y_train)

# Karar sınırlarını görselleştirmek için meshgrid oluştur
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Meshgrid üzerindeki her noktanın sınıfını tahmin et
Z = cat_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Karar sınırlarını çizdir
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)

# Eğitim ve test verilerini scatter plot olarak çizdir
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', label='Eğitim Verileri', edgecolor='k')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='o', label='Test Verileri', edgecolor='k')

# Grafik ayarları
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])
plt.title('CatBoost Karar Sınırları (İris Veriseti)')
plt.legend()

# Grafik kaydetme
plt.savefig('C:/Users/firat/source/repos/FiratMLDersleri/Kaydedilenler/catboost_decision_boundaries.png')

# catboost_info klasörünü taşımak için
source_path = 'catboost_info'
destination_path = 'C:/Users/firat/source/repos/FiratMLDersleri/Kaydedilenler/catboost_info'

# Klasör var mı kontrol et, yoksa taşı
if os.path.exists(source_path):
    shutil.move(source_path, destination_path)

# Grafiği göster
plt.show()
