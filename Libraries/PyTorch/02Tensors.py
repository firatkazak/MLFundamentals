import torch
import numpy as np

# Tensörler, dizilere ve matrislere çok benzeyen özel bir veri yapısıdır. PyTorch'ta, bir modelin girdilerini ve çıktılarını ve ayrıca modelin parametrelerini kodlamak için tensörleri kullanırız.
#
# Tensörler NumPy'nin ndarrays'ine benzer, ancak tensörler GPU'larda veya diğer donanım hızlandırıcılarında çalışabilir.
# Aslında, tensörler ve NumPy dizileri genellikle aynı temel belleği paylaşabilir ve veri kopyalama ihtiyacını ortadan kaldırır.
# Tensörler ayrıca otomatik farklılaştırma için optimize edilmiştir.

# Bir Tensörü Initialize Etme: Tensörler çeşitli şekillerde başlatılabilir(Initialize edilebilir).

# Doğrudan veriden: Tensörler doğrudan verilerden oluşturulabilir. Veri türü otomatik olarak çıkarılır;
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# Bir NumPy dizisinden: Tensörler NumPy dizilerinden oluşturulabilir;
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# Başka bir tensörden: Yeni tensör, açıkça geçersiz kılınmadığı sürece, argüman tensörünün özelliklerini (şekil, veri türü) korur;
x_ones = torch.ones_like(x_data)  # x_data'nın özelliklerini korur.
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # x_data'nın veri türünü geçersiz kılar.
print(f"Random Tensor: \n {x_rand} \n")

# Rastgele veya sabit değerlerle: shape, tensör boyutlarının bir tuple'ıdır. Aşağıdaki fonksiyonlarda, çıktı tensörünün boyutluluğunu belirler;
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# Bir Tensörün Nitelikleri: Tensör öznitelikleri şekillerini, veri türlerini ve depolandıkları cihazı tanımlar;
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# Tensörler Üzerinde İşlemler;
# Aritmetik, lineer cebir, matris manipülasyonu (transpoze etme, indeksleme, dilimleme), örnekleme ve daha fazlası dahil olmak üzere 100'den fazla tensör işlemi burada kapsamlı bir şekilde açıklanmaktadır.
# https://pytorch.org/docs/stable/torch.html
# Bu işlemlerin her biri GPU üzerinde çalıştırılabilir (CPU'ya göre tipik olarak daha yüksek hızlarda).
# Colab kullanıyorsanız, Çalışma Zamanı > Çalışma zamanı türünü değiştir > GPU'ya giderek bir GPU tahsis edin.
# Varsayılan olarak, tensörler CPU üzerinde oluşturulur.
# Tensörleri .to yöntemini kullanarak açıkça GPU'ya taşımamız gerekir (GPU kullanılabilirliğini kontrol ettikten sonra).
# Büyük tensörleri cihazlar arasında kopyalamanın zaman ve bellek açısından pahalı olabileceğini unutmayın!

# Varsa tensörümüzü GPU'ya taşıyoruz;
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

# Standart numpy benzeri indeksleme ve dilimleme;
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:, 1] = 0
print(tensor)

# Tensörleri birleştirme: Belirli bir boyut boyunca bir dizi tensörü birleştirmek için torch.cat kullanabilirsiniz.
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
# NOT: Ayrıca torch.cat'ten biraz farklı olan başka bir tensör birleştirme operatörü olan torch.stack'e de bakın.

# Aritmetik işlemler
# Bu, iki tensör arasındaki matris çarpımını hesaplar. y1, y2, y3 aynı değere sahip olacaktır
# ``tensor.T`` bir tensörün transpozesini döndürür
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

# Bu, eleman-bilge çarpımını hesaplar. z1, z2, z3 aynı değere sahip olacaktır
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# Tek elemanlı tensörler: Tek elemanlı bir tensörünüz varsa, örneğin bir tensörün tüm değerlerini tek bir değerde toplayarak, item() kullanarak bunu bir Python sayısal değerine dönüştürebilirsiniz:
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# Yerinde işlemler: Sonucu işlenene depolayan işlemlere yerinde işlemler denir. Bunlar _ sonekiyle gösterilir. Örneğin: x.copy_(y), x.t_(), x'i değiştirecektir.
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)
# NOT: Yerinde işlemler bellekten bir miktar tasarruf sağlar, ancak türevleri hesaplarken geçmişin anında kaybedilmesi nedeniyle sorunlu olabilir. Bu nedenle, kullanımları tavsiye edilmez.

# NumPy ile köprü: CPU ve NumPy dizilerindeki tensörler temel bellek konumlarını paylaşabilir ve birini değiştirmek diğerini de değiştirir.

# Tensörden NumPy dizisine;
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
# Tensördeki bir değişiklik NumPy dizisine yansır;
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# NumPy dizisinden Tensöre;
n = np.ones(5)
t = torch.from_numpy(n)
# NumPy dizisindeki değişiklikler tensöre yansır;
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
