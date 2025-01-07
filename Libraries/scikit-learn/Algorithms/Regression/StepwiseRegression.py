import numpy as np  # NumPy kütüphanesini import ediyoruz.
import pandas as pd  # Pandas kütüphanesini veri işleme için import ediyoruz.
import statsmodels.api as sm  # statsmodels kütüphanesini istatistiksel modelleme için import ediyoruz.
import matplotlib.pyplot as plt  # Matplotlib kütüphanesinin pyplot modülünü import ediyoruz.

# Örnek veri oluşturma
np.random.seed(0)  # Sonuçların tekrar edilebilir olmasını sağlamak için rastgele sayı tohumunu ayarlıyoruz.
X = np.random.rand(100, 3)  # 100 gözlem ve 3 bağımsız değişken oluşturuyoruz.
y = 2 + 3 * X[:, 0] + 4 * X[:, 1] + np.random.randn(100)  # Y bağımlı değişken; X'e bağlı bir ilişki oluşturuyoruz.

# Veriyi DataFrame'e çeviriyoruz
df = pd.DataFrame(X, columns=['Feature1', 'Feature2', 'Feature3'])  # Bağımsız değişkenler için bir DataFrame oluşturuyoruz.
df['Target'] = y  # Bağımlı değişkeni ekliyoruz.

# DataFrame'in yapısını kontrol et
print(df.head())  # DataFrame'in ilk 5 satırını yazdırıyoruz.


# Stepwise Regression fonksiyonu
def stepwise_regression(data, target):  # Stepwise regresyonu uygulamak için bir fonksiyon tanımlıyoruz.
    initial_features = data.columns.tolist()  # Başlangıçta tüm özellikleri listeye alıyoruz.
    selected_features = []  # Seçilen özellikler listesini başlatıyoruz.

    while initial_features:  # Başlangıç özellikleri varken döngü devam eder.
        remaining_features = list(set(initial_features) - set(selected_features))  # Kalan özellikleri alıyoruz.
        new_pval = pd.Series(index=remaining_features)  # Yeni p değerlerini saklamak için boş bir seri oluşturuyoruz.

        for feature in remaining_features:  # Kalan özellikler için döngü
            model = sm.OLS(data[target], sm.add_constant(data[selected_features + [feature]])).fit()  # OLS regresyonu oluşturuyoruz.
            new_pval[feature] = model.pvalues[feature]  # Özelliğin p değerini alıyoruz.

        min_pval_feature = new_pval.idxmin()  # En küçük p değerine sahip özelliği buluyoruz.
        if new_pval[min_pval_feature] < 0.05:  # Eğer p değeri 0.05'ten küçükse
            selected_features.append(min_pval_feature)  # Özelliği seçilen özellikler listesine ekliyoruz.
            initial_features.remove(min_pval_feature)  # Başlangıç özelliklerinden çıkarıyoruz.
        else:
            break  # Eğer p değeri 0.05'ten büyükse döngüyü sonlandırıyoruz.

    return selected_features  # En iyi özellikler listesini döndürüyoruz.


# Stepwise Regression uygulama
target_variable = 'Target'  # Hedef değişkeni belirtiyoruz
print("Target variable: ", target_variable)  # Hedef değişkeni kontrol ediyoruz.
best_features = stepwise_regression(df, target_variable)  # Stepwise regresyonu uyguluyoruz.

# Sonuçları görselleştirme
# Gerçek değerleri mavi ile gösteriyoruz;
plt.scatter(df['Feature1'], df['Target'], color='blue', label='Gerçek Değerler')
# Tahmin edilen değerleri kırmızı ile gösteriyoruz;
plt.scatter(df['Feature1'], sm.OLS(df[target_variable], sm.add_constant(df[best_features])).fit().predict(), color='red', label='Tahmin Edilen Değerler')
plt.xlabel('Feature1')  # X ekseni için etiket
plt.ylabel('Target')  # Y ekseni için etiket
plt.title('Stepwise Regression')  # Grafiğin başlığını belirtiyoruz.
plt.legend()  # Grafikteki etiketler için bir efsane oluşturuyoruz.
plt.show()  # Grafiği gösteriyoruz.

# Stepwise Regression
# Stepwise regression, bir modelin hangi bağımsız değişkenlerin hedef değişkenle en iyi ilişkiyi kurduğunu belirlemek için kullanılan bir modelleme tekniğidir.
# Stepwise regression, modele adım adım değişken ekler veya çıkarır. Bu genellikle özellik seçimi için kullanılır.

# Stepwise Regresyonun Amacı

# Özellik Seçimi:
# Stepwise regresyon, veri setindeki birçok bağımsız değişkenden hangilerinin hedef değişken üzerinde anlamlı bir etkisi olduğunu belirlemek için kullanılır.
# Bu, modelin karmaşıklığını azaltır ve aşırı öğrenmeyi (overfitting) engelleyebilir.

# Anlamlı Değişkenler:
# Bu yöntem, yalnızca anlamlı p-değerine sahip bağımsız değişkenleri modele dahil ederek, modelin daha yorumlanabilir olmasını sağlar.
# Genellikle p-değeri 0.05 veya daha düşük olan değişkenler anlamlı kabul edilir.

# Modeli Optimize Etme:
# Stepwise regresyon, modelin genel performansını artırmak için gerekli olan en az sayıda değişkenle çalışır.
# Bu, gereksiz değişkenleri dışarıda bırakarak daha sade bir model elde etmemizi sağlar.
