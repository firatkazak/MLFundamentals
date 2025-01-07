import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Data: Experience (years) and whether they got promoted (1 = Yes, 0 = No)
X = np.array([[1], [2], [3], [4], [5], [6]])  # Independent variable: experience
y = np.array([0, 0, 0, 1, 1, 1])  # Dependent variable: promotion (yes/no)

# Create the logistic regression model with regularization (L2 is default)
model = LogisticRegression(C=1.0,
                           penalty='l2'
                           )
# C: C, düzenlemenin (regularization) gücünü kontrol eden bir değerdir. C değeri küçüldükçe, model daha fazla düzenlenir.
# C: Bu, modelin daha basit hale gelmesi ve aşırı öğrenmeyi (overfitting) önlemeye yardımcı olması anlamına gelir. Ancak, modelin veriye uyum sağlama yeteneği de azalabilir.
# C değeri büyüdükçe, model daha az düzenlenir. Bu, modelin veriye daha iyi uyum sağlamasına olanak tanır ancak aşırı öğrenme riskini artırır.
# C değeri genelde 1 alınır.

# Penalty: Penalty parametresi, modelin düzenlenmesinde kullanılan ceza terimini belirtir.

# L2 Düzenleme: penalty='l2' seçildiğinde, L2 normu (Ridge regresyonu) kullanılarak düzenleme yapılır.
# L2 düzenleme, tüm katsayıların küçük değerler almasını teşvik eder ve böylece modelin daha basit hale gelmesini sağlar.

# L1 Düzenleme: penalty='l1' seçildiğinde, L1 normu (Lasso regresyonu) kullanılarak düzenleme yapılır.
# L1 düzenleme, bazı katsayıların sıfıra eşit olmasına neden olabilir ve bu da özellik seçimi (feature selection) yapmaya yardımcı olur.

# Diğer Düzenleme Türleri: Bazı kütüphaneler, ElasticNet gibi diğer düzenleme türlerini de destekler.

model.fit(X, y)

# Make predictions
y_pred = model.predict(X)  # Tahmin
y_prob = model.predict_proba(X)  # Olasılık

print("Predicted Promotions: ", y_pred)
print("Prediction Probabilities: ", y_prob)

# Görselleştirme için yeni X değerleri (daha fazla veri noktası ile)
X_new = np.linspace(start=0, stop=7, num=300).reshape(-1, 1)

# Tahmin edilen olasılıklar (sınıf 1 olma olasılığı)
y_prob_new = model.predict_proba(X_new)[:, 1]

# Verileri çizdir
plt.scatter(X, y, color='black', label='Gerçek Veriler')

# Logistic Regression eğrisini çizdir
plt.plot(X_new, y_prob_new, color='blue', label='Lojistik Regresyon Eğrisi (L2 Düzenleme)')

# Grafiği ayarla
plt.xlabel('Deneyim (Yıl)')
plt.ylabel('Terfi Olasılığı')
plt.title('Lojistik Regresyon: Deneyim ve Terfi Olasılığı (L2 Düzenleme ile)')
plt.legend()
plt.show()

# Regularization ve Görselleştirme:

# Regularization (C): Küçük bir C değeri modelin esnekliğini sınırlar ve overfitting’i önler.

# Eğri Görselleştirmesi: L2 regularization, modelin karar sınırını daha düzgün ve dengeli hale getirir.
# Küçük veri setlerinde bu, genellikle daha genellenebilir sonuçlar verir.

# Bu görselleştirme, deneyime dayalı terfi olasılıklarını regularization ile nasıl tahmin ettiğimizi gösterir.
# Regularization’ın etkisini, eğrinin aşırı fit olmadan daha düzgün çizildiği bir grafikle gözlemleyebiliriz.
