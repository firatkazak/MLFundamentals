import pandas as pd  # pandas kütüphanesini veri analizi için içe aktarıyoruz
from pandas.api.types import CategoricalDtype  # Kategorik veri türü tanımlamak için CategoricalDtype içe aktarılıyor
import matplotlib.pyplot as plt  # Veri görselleştirme için matplotlib.pyplot kütüphanesi içe aktarılıyor
from statsmodels.miscmodels.ordinal_model import OrderedModel  # Sıralı regresyon analizi için OrderedModel içe aktarılıyor

# diamonds.csv dosyasını okur ve data_diam adında bir DataFrame oluşturur
data_diam = pd.read_csv("C:/Users/firat/source/repos/FiratMLDersleri/Kaydedilenler/diamonds.csv")
print(data_diam.head(10))  # İlk 10 satırı ekrana yazdırır
print(data_diam.dtypes)  # Her sütunun veri tiplerini gösterir

# Kategorik veri türü tanımlanıyor, 'cut' sütunu için sıralı kategoriler belirleniyor
cat_type = CategoricalDtype(categories=['Fair', 'Good', 'Ideal', 'Very Good', 'Premium'], ordered=True)
# 'cut' sütunu belirtilen kategorik türüne dönüştürülüyor
datadiam = data_diam["cut"] = data_diam["cut"].astype(cat_type)
print("data diam:", datadiam)  # Dönüştürülen kategorik veriyi yazdırır

# 'volume' adında yeni bir sütun oluşturur; hacmi hesaplamak için x, y ve z çarpılıyor
data_diam['volume'] = data_diam['x'] * data_diam['y'] * data_diam['z']
# 'x', 'y', 'z' sütunları DataFrame'den siliniyor
data_diam.drop(['x', 'y', 'z'], axis=1, inplace=True)

# Grafik boyutunu ayarlıyoruz
plt.figure(figsize=[24, 24])
plt.subplot(221)  # İlk alt alanı seçiyoruz
plt.hist(data_diam['carat'], bins=20, color='b')  # 'carat' için histogram çiziliyor
plt.xlabel('Weight')  # X eksenine etiket ekleniyor
plt.title('Distribution by Weight')  # Histogram başlığı ayarlanıyor
plt.subplot(222)  # İkinci alt alanı seçiyoruz
plt.hist(data_diam['depth'], bins=20, color='r')  # 'depth' için histogram çiziliyor
plt.xlabel('Diamond Depth')  # X eksenine etiket ekleniyor
plt.title('Distribution by Depth')  # Histogram başlığı ayarlanıyor
plt.subplot(223)  # Üçüncü alt alanı seçiyoruz
plt.hist(data_diam['price'], bins=20, color='g')  # 'price' için histogram çiziliyor
plt.xlabel('Price')  # X eksenine etiket ekleniyor
plt.title('Distribution by Price')  # Histogram başlığı ayarlanıyor
plt.subplot(224)  # Dördüncü alt alanı seçiyoruz
plt.hist(data_diam['volume'], bins=20, color='m')  # 'volume' için histogram çiziliyor
plt.xlabel('Volume')  # X eksenine etiket ekleniyor
plt.title('Distribution by Volume')  # Histogram başlığı ayarlanıyor
plt.show()  # Tüm grafikleri göster

# Probit dağılımı ile sıralı regresyon modeli oluşturuluyor
mod_prob = OrderedModel(data_diam['cut'], data_diam[['volume', 'price', 'carat']], distr='probit')
# Model fit ediliyor
res_prob = mod_prob.fit(method='bfgs')
res_prob.summary()  # Modelin özetini göster

# Logit dağılımı ile sıralı regresyon modeli oluşturuluyor
mod_prob = OrderedModel(data_diam['cut'], data_diam[['volume', 'price', 'carat']], distr='logit')
# Model fit ediliyor
res_log = mod_prob.fit(method='bfgs')
res_log.summary()  # Modelin özetini göster

# Bağımsız değişkenler kullanılarak tahmin yapılıyor
predicted = res_log.model.predict(res_log.params, exog=data_diam[['volume', 'price', 'carat']])
print(predicted)  # Tahmin sonuçlarını yazdır

# Ordinal regresyon, bağımlı değişkenin sıralı kategorilere sahip olduğu durumlarda kullanılan bir regresyon tekniğidir.
# Bu tür regresyon, bağımlı değişkenin sadece kategorik değil, aynı zamanda sıralı olduğu durumlarda (örneğin, "kötü", "orta", "iyi" gibi) tercih edilir.

# Temel Kavramlar
# Bağımlı Değişken: Sıralı kategorilere sahip olan değişkendir. Örneğin, bir ürün değerlendirmesi "kötü", "orta", "iyi" gibi değerler alabilir.
# Bağımsız Değişkenler: Bağımlı değişkeni etkileyen faktörlerdir. Sürekli veya kategorik olabilir.
# Sıralı Kategoriler: Kategoriler arasında bir sıralama bulunur; yani, bir kategori diğerine göre daha yüksek veya daha düşük bir değere sahiptir.

# Kullanım Alanları
# Müşteri memnuniyeti anketleri
# Sağlık durumunun değerlendirilmesi (örneğin, hafif, orta, ağır)
# Eğitimde başarı seviyeleri (örneğin, yeterli, iyi, çok iyi)

# Ordinal Regresyon Modelleri
# Probit Regresyon: Sıralı kategoriler arasındaki olasılıkları tahmin etmek için kullanılır.
# Logit Regresyon: Ordinal bağımlı değişkenin her bir kategorisinin olasılığını tahmin eder.
