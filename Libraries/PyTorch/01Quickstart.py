import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# PyTorch'un verilerle çalışmak için iki ilkeli vardır: torch.utils.data.DataLoader ve torch.utils.data.Dataset.
# Dataset örnekleri ve bunlara karşılık gelen etiketleri saklar ve DataLoader Dataset'in etrafına bir iterable sarar.
# PyTorch, hepsi veri kümeleri içeren TorchText, TorchVision ve TorchAudio gibi alana özgü kütüphaneler sunar. Bu eğitim için bir TorchVision veri kümesi kullanacağız.
# Torchvision.datasets modülü, CIFAR, COCO gibi birçok gerçek dünya görüntü verisi için Veri Kümesi nesneleri içerir. Bu eğitimde FashionMNIST veri kümesini kullanacağız.
# Her TorchVision Veri Kümesi iki argüman içerir: sırasıyla örnekleri ve etiketleri değiştirmek için transform ve target_transform.

# Açık veri kümelerinden eğitim verilerini indirme;
training_data = datasets.FashionMNIST(
    root="C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Downloaded/MNISTdata",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Açık veri kümelerinden test verilerini indirme.
test_data = datasets.FashionMNIST(
    root="C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Downloaded/MNISTdata",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Veri Kümesini DataLoader'a bir argüman olarak iletiyoruz.
# Bu, veri kümemiz üzerinde bir yinelenebilir dosyayı sarar ve otomatik gruplama, örnekleme, karıştırma ve çok işlemli veri yüklemeyi destekler.
# Burada 64'lük bir yığın boyutu tanımlıyoruz, yani dataloader yinelenebilirindeki her öğe 64 özellik ve etiketten oluşan bir yığın döndürecektir.

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Model Oluşturma
# PyTorch'ta bir sinir ağı tanımlamak için nn.Module'den miras alan bir sınıf oluşturuyoruz.
# Ağın katmanlarını __init__ fonksiyonunda tanımlarız ve verilerin forward fonksiyonunda ağdan nasıl geçeceğini belirtiriz.
# Sinir ağındaki işlemleri hızlandırmak için, varsa GPU'ya veya MPS'ye taşıyoruz.

# Eğitim için cpu, gpu veya mps cihazı temin eder.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Model tanımlama
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=28 * 28, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

# Model Parametrelerinin Optimize Edilmesi: Bir modeli eğitmek için bir kayıp fonksiyonuna ve bir optimize ediciye ihtiyacımız vardır.
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# Tek bir eğitim döngüsünde, model eğitim veri kümesi üzerinde tahminler yapar (toplu olarak beslenir) ve modelin parametrelerini ayarlamak için tahmin hatasını geriye doğru yayar.

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# Ayrıca, modelin performansını test veri setine göre kontrol ederek öğrendiğinden emin oluyoruz.

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# Eğitim süreci birkaç iterasyon (epok) üzerinden yürütülür. Her dönem boyunca model daha iyi tahminler yapmak için parametreleri öğrenir.
# Modelin her dönemdeki doğruluğunu ve kaybını yazdırırız; her dönemde doğruluğun arttığını ve kaybın azaldığını görmek isteriz.

epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# Modeli Kaydetme: Bir modeli kaydetmenin yaygın bir yolu, dahili durum sözlüğünü (model parametrelerini içeren) serileştirmektir;
torch.save(model.state_dict(), f="C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Downloaded/MNISTmodel")
print("Saved PyTorch Model State to model.pth")

# Model Yükleme: Bir modelin yüklenmesi süreci, model yapısının yeniden oluşturulmasını ve durum sözlüğünün içine yüklenmesini içerir;
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load(f="C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Downloaded/MNISTmodel", weights_only=True))

# Bu model artık tahminlerde bulunmak için kullanılabilir;
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

# Sonuç: Çıktı olarak Predicted: "Ankle boot", Actual: "Ankle boot" yazdı. Tahmin ankle boot'muş.