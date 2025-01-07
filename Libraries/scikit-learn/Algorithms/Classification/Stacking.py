import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Iris veri setini yükle
data = load_iris()
X = data.data
y = data.target

# Eğitim ve test setine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Temel modelleri tanımla
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=0)),
    ('gb', GradientBoostingClassifier(random_state=0))
]

# Yığınlama sınıflandırıcısını oluştur
stacking_model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression())
stacking_model.fit(X_train, y_train)

# Tahmin yap
y_pred = stacking_model.predict(X_test)

# Sonuçları yazdır
print("Doğruluk Skoru:", accuracy_score(y_test, y_pred))

# Sonuçları görselleştir
# İki özellik için görselleştirme yapacağız (özellik 1 ve 2)
plt.figure(figsize=(10, 6))
colors = ['red', 'green', 'blue']
markers = ['o', 's', 'x']

# Eğitim verilerini ve test verilerini görselleştir
for i, color in zip(np.unique(y), colors):
    plt.scatter(X_train[y_train == i, 0], X_train[y_train == i, 1],
                color=color, alpha=0.5, label=f'Eğitim Sınıf {i}', marker=markers[i])
    # Test verilerini dolu işaretçilerle göster
    plt.scatter(X_test[y_test == i, 0], X_test[y_test == i, 1],
                color=color, s=100, label=f'Test Sınıf {i}', marker='o', edgecolor=color)

plt.title('Stacking Sonuçları')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.legend()
plt.grid()
plt.show()
