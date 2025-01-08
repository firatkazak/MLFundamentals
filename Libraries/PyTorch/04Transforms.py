import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

# Dönüşümler
# Veriler her zaman makine öğrenimi algoritmalarını eğitmek için gerekli olan işlenmiş son haliyle gelmez.
# Veriler üzerinde bazı manipülasyonlar gerçekleştirmek ve eğitim için uygun hale getirmek için dönüşümler kullanırız.
# Tüm TorchVision veri kümeleri, dönüşüm mantığını içeren çağrılabilirleri kabul eden iki parametreye sahiptir;
# özellikleri değiştirmek için transform ve etiketleri değiştirmek için target_transform-.
# torchvision.transforms modülü, yaygın olarak kullanılan birkaç dönüşümü kutudan çıkarır.
# FashionMNIST özellikleri PIL Image formatındadır ve etiketler tam sayıdır.
# Eğitim için, özelliklerin normalize edilmiş tensörler ve etiketlerin de one-hot kodlu tensörler olması gerekir.
# Bu dönüşümleri yapmak için ToTensor ve Lambda kullanıyoruz.

ds = datasets.FashionMNIST(
    root="C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Downloaded/MNISTdata",
    train=True,
    download=True,
    transform=ToTensor(),  # ToTensor, bir PIL görüntüsünü veya NumPy ndarray'ini bir FloatTensor'a dönüştürür. ve görüntünün piksel yoğunluk değerlerini [0., 1.] aralığında ölçeklendirir.
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

# Lambda Transforms
# Lambda dönüşümleri kullanıcı tanımlı herhangi bir lambda fonksiyonunu uygular.
# Burada, tamsayıyı tek vuruşla kodlanmış bir tensöre dönüştürmek için bir fonksiyon tanımlıyoruz.
# İlk olarak 10 boyutunda (veri setimizdeki etiket sayısı) bir sıfır tensörü oluşturur ve y etiketi tarafından verilen indekse = 1 değerini atayan scatter_ fonksiyonunu çağırır.
target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
