import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Data: Experience (years) and whether they got promoted (1 = Yes, 0 = No)
X = np.array([[1], [2], [3], [4], [5], [6]])  # Independent variable: Deneyim.
y = np.array([0, 0, 0, 1, 1, 1])  # Dependent variable: Terfi ihtimali.
# Terfi ihtimalini(y) ölçeceğiz. Terfi ihtimalimiz, deneyimimize bağımlı.

model = LogisticRegression()  # 1 tane logistic regression modeli oluşturuyoruz.
model.fit(X, y)  # Bu modele bağımsız ve bağımlı değişkenimizi vererek eğitiyoruz.

# Make predictions
y_pred = model.predict(X)  # modelin X verileri üzerindeki tahminlerini alıp y_pred değişkenine atıyoruz.
y_prob = model.predict_proba(X)  # modelin X verileri üzerindeki tahminlerinin sınıf olasılıklarını alıp y_prob değişkenine atıyoruz.

print("Predicted Promotions: ", y_pred)  # Tahminleri ekrana yazdırıyoruz.
print("Prediction Probabilities: ", y_prob)  # Olasılıkları ekrana yazdırıyoruz.

# Görselleştirme için yeni X değerleri (daha fazla veri noktası ile)
X_new = np.linspace(start=0, stop=7, num=300).reshape(-1, 1)

# Tahmin edilen olasılıklar (sınıf 1 olma olasılığı)
y_prob_new = model.predict_proba(X_new)[:, 1]

# Verileri çizdir
plt.scatter(X, y, color='black', label='Gerçek Veriler')

# Logistic Regression eğrisini çizdir
plt.plot(X_new, y_prob_new, color='blue', label='Lojistik Regresyon Eğrisi')

# Grafiği ayarla
plt.xlabel('Deneyim (Yıl)')
plt.ylabel('Terfi Olasılığı')
plt.title('Lojistik Regresyon: Deneyim ve Terfi Olasılığı')
plt.legend()
plt.show()

# Doğrusal regresyon, ilgili ve bilinen başka bir veri değeri kullanarak bilinmeyen verilerin değerini tahmin eden bir veri analizi tekniğidir.
# Bilinmeyen veya bağımlı değişkeni ve bilinen veya bağımsız değişkeni doğrusal bir denklem olarak matematiksel olarak modeller.
# Örneğin, geçen yılki harcamalarınız ve geliriniz hakkında verileriniz olduğunu varsayalım.
# Doğrusal regresyon teknikleri bu verileri analiz eder ve giderlerinizin gelirinizin yarısı olduğunu belirler.
# Daha sonra gelecekteki bilinen bir geliri yarıya indirerek bilinmeyen bir gelecekteki gideri hesaplarlar.
# Ekonomideki ceteris paribus örnek verilebilir.