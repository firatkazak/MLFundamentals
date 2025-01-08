import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Sinir Ağını Oluşturma
# Sinir ağları, veriler üzerinde işlem yapan katmanlardan/modüllerden oluşur.
# torch.nn isim alanı, kendi sinir ağınızı oluşturmak için ihtiyacınız olan tüm yapı taşlarını sağlar.
# PyTorch'taki her modül nn.Module alt sınıfına sahiptir. Bir sinir ağı, diğer modüllerden (katmanlardan) oluşan bir modüldür.
# Bu iç içe geçmiş yapı, karmaşık mimarilerin kolayca oluşturulmasına ve yönetilmesine olanak tanır.

# Eğitim için Cihaz Edinme: Modelimizi, varsa GPU veya MPS gibi bir donanım hızlandırıcı üzerinde eğitebilmek istiyoruz. Torch.cuda veya torch.backends.mps'nin kullanılabilir olup olmadığını kontrol edelim, aksi takdirde CPU'yu kullanırız.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Class Tanımlama
# Sinir ağımızı nn.Module alt sınıfını kullanarak tanımlarız ve sinir ağı katmanlarını __init__ içinde başlatırız.
# Her nn.Module alt sınıfı, forward yönteminde girdi verileri üzerindeki işlemleri uygular.

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=28 * 28, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# NeuralNetwork'ün bir örneğini oluşturup cihaza taşıyoruz ve yapısını yazdırıyoruz.
model = NeuralNetwork().to(device)
print(model)

# Modeli kullanmak için ona girdi verilerini iletiriz.
# Bu, bazı arka plan işlemleriyle birlikte modelin ileriye doğru çalışmasını sağlar. model.forward() işlevini doğrudan çağırmayın!
# Modelin girdi üzerinde çağrılması, her sınıf için 10 ham tahmin değerinin her çıktısına karşılık gelen dim=0 ve her çıktının tek tek değerlerine karşılık gelen dim=1 olan 2 boyutlu bir tensör döndürür.
# Tahmin olasılıklarını nn.Softmax modülünün bir örneğinden geçirerek elde ederiz.
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

# Model Katmanları
# FashionMNIST modelindeki katmanları inceleyelim.
# Bunu göstermek için, 28x28 boyutunda 3 görüntüden oluşan örnek bir mini parti alacağız ve ağdan geçirirken ona ne olduğunu göreceğiz.
input_image = torch.rand(3, 28, 28)
print(input_image.size())

# nn.Flatten: Her bir 2D 28x28 görüntüyü 784 piksel değerinden oluşan bitişik bir diziye dönüştürmek için nn.Flatten katmanını başlatıyoruz (minibatch boyutu (dim=0'da) korunur).
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# nn.Linear: Doğrusal katman, depolanan ağırlıkları ve önyargıları kullanarak girişe doğrusal bir dönüşüm uygulayan bir modüldür.
layer1 = nn.Linear(in_features=28 * 28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# nn.ReLU: Doğrusal olmayan aktivasyonlar, modelin girdileri ve çıktıları arasındaki karmaşık eşleşmeleri yaratan şeydir.
# Doğrusal dönüşümlerden sonra uygulanarak doğrusal olmama özelliği kazandırırlar ve sinir ağlarının çok çeşitli olguları öğrenmesine yardımcı olurlar.
# Bu modelde, doğrusal katmanlarımız arasında nn.ReLU kullanıyoruz, ancak modelinize doğrusal olmama özelliği kazandırmak için başka aktivasyonlar da var.
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# nn.Sequential: nn.Sequential, modüllerin sıralı bir konteyneridir.
# Veriler tüm modüllerden tanımlandığı sırayla geçirilir. Seq_modules gibi hızlı bir ağ oluşturmak için sıralı kapsayıcılar kullanabilirsiniz.
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(in_features=20, out_features=10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)

# nn.Softmax: Sinir ağının son doğrusal katmanı, nn.Softmax modülüne aktarılan logitleri - [-infty, infty] cinsinden ham değerler - döndürür.
# Logitler, modelin her sınıf için öngörülen olasılıklarını temsil eden [0, 1] değerlerine ölçeklendirilir.
# dim parametresi, değerlerin toplamının 1 olması gereken boyutu belirtir.
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

# Model Parametreleri
# Bir sinir ağı içindeki birçok katman parametrelendirilmiştir, yani eğitim sırasında optimize edilen ilişkili ağırlıklara ve önyargılara sahiptir.
# nn.Module alt sınıfı, model nesnenizin içinde tanımlanan tüm alanları otomatik olarak izler ve modelinizin parameters() veya named_parameters() yöntemlerini kullanarak tüm parametreleri erişilebilir hale getirir.
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
