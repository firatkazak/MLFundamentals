import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Iris veri setini yükle
data = load_iris()
X = data.data
y = data.target

# Eğitim ve test setine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# XGBoost modelini oluştur
xgb_model = xgb.XGBClassifier(eval_metric='mlogloss')  # use_label_encoder kaldırıldı
xgb_model.fit(X_train, y_train)

# Tahmin yap
y_pred = xgb_model.predict(X_test)

# Sonuçları yazdır
print("Doğruluk Skoru:", accuracy_score(y_test, y_pred))

# Sonuçları görselleştir
plt.figure(figsize=(10, 6))

# Eğitim ve test verilerini görselleştir
scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='o', edgecolor='k', s=100, label='Test Verileri')
plt.title('XGBoost Sınıflandırma Sonuçları')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.colorbar(scatter, label='Sınıf')
plt.legend()
plt.grid()
plt.show()

# XGBoost
# Alegori: Farklı türdeki verileri işlerken hızlı ve etkili bir yöntem sunar.
# Karar ağaçlarını bir araya getirerek güçlü bir model oluşturur.
