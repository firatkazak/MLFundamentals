import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Model Parametrelerinin Optimize Edilmesi
# Artık bir modelimiz ve verilerimiz olduğuna göre, parametrelerini verilerimiz üzerinde optimize ederek modelimizi eğitme, doğrulama ve test etme zamanı gelmiştir.
# Bir modeli eğitmek iteratif bir süreçtir; her iterasyonda model çıktı hakkında bir tahminde bulunur, tahminindeki hatayı (kayıp) hesaplar, hatanın parametrelerine göre türevlerini toplar (önceki bölümde gördüğümüz gibi) ve bu parametreleri gradyan inişi kullanarak optimize eder.

training_data = datasets.FashionMNIST(
    root="C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Downloaded/MNISTdata",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Downloaded/MNISTdata",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


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


model = NeuralNetwork()

# Hiperparametreler
# Hiperparametreler, model optimizasyon sürecini kontrol etmenizi sağlayan ayarlanabilir parametrelerdir.
# Farklı hiperparametre değerleri model eğitimini ve yakınsama oranlarını etkileyebilir.
# Eğitim için aşağıdaki hiperparametreleri tanımlıyoruz:
# Number of Epochs: Veri kümesi üzerinde yinelenecek sayı.
# Batch Size: Parametreler güncellenmeden önce ağ boyunca yayılan veri örneklerinin sayısı.
# Learning Rate: Her grupta/epokta model parametrelerinin ne kadar güncelleneceği. Daha küçük değerler yavaş öğrenme hızı sağlarken, büyük değerler eğitim sırasında öngörülemeyen davranışlara neden olabilir.

learning_rate = 1e-3
batch_size = 64
epochs = 5

# Optimizasyon Döngüsü
# Hiperparametrelerimizi ayarladıktan sonra, modelimizi bir optimizasyon döngüsü ile eğitebilir ve optimize edebiliriz.
# Optimizasyon döngüsünün her iterasyonuna bir epok denir.
# Her epok iki ana bölümden oluşur:
# The Train Loop: Eğitim veri kümesi üzerinde yineleme yapar ve optimum parametrelere yakınsamaya çalışır.
# The Validation/Test Loop: Model performansının iyileşip iyileşmediğini kontrol etmek için test veri kümesi üzerinde yineleme yapın.
# Eğitim döngüsünde kullanılan bazı kavramları kısaca tanıyalım. Optimizasyon döngüsünün Tam Uygulamasını görmek için ileri atlayın.

# Loss Function(Kayıp Fonksiyonu)
# Bazı eğitim verileri sunulduğunda, eğitimsiz ağımızın doğru cevabı vermemesi muhtemeldir.
# Kayıp fonksiyonu, elde edilen sonucun hedef değere benzememe derecesini ölçer ve eğitim sırasında en aza indirmek istediğimiz kayıp fonksiyonudur.
# Kaybı hesaplamak için, verilen veri örneğimizin girdilerini kullanarak bir tahmin yaparız ve bunu gerçek veri etiketi değeriyle karşılaştırırız.
# Yaygın kayıp fonksiyonları arasında regresyon görevleri için nn.MSELoss (Ortalama Hata Karesi) ve sınıflandırma için nn.NLLLoss (Negatif Log Olabilirlik) bulunur.
# nn.CrossEntropyLoss, nn.LogSoftmax ve nn.NLLLoss'u birleştirir.
# Modelimizin çıktı logitlerini, logitleri normalleştirecek ve tahmin hatasını hesaplayacak olan nn.CrossEntropyLoss'a aktarıyoruz.

loss_fn = nn.CrossEntropyLoss()

# Optimizer(Optimize Edici)
# Optimizasyon, her eğitim adımında model hatasını azaltmak için model parametrelerini ayarlama sürecidir.
# Optimizasyon algoritmaları bu işlemin nasıl gerçekleştirildiğini tanımlar (bu örnekte Stokastik Gradyan İnişi kullanıyoruz).
# Tüm optimizasyon mantığı optimizer nesnesi içinde kapsüllenmiştir. Burada SGD optimize edicisini kullanıyoruz;
# ayrıca PyTorch'ta ADAM ve RMSProp gibi farklı model ve veri türleri için daha iyi çalışan birçok farklı optimize edici mevcuttur.

# Eğitilmesi gereken model parametrelerini kaydederek ve öğrenme oranı hiper parametresini girerek optimize ediciyi başlatıyoruz.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# Eğitim döngüsü içinde optimizasyon üç adımda gerçekleşir:
# Model parametrelerinin gradyanlarını sıfırlamak için optimizer.zero_grad() işlevini çağırın.
# Gradyanlar varsayılan olarak toplanır; çift sayımı önlemek için her iterasyonda açıkça sıfırlarız.
# Tahmin kaybını loss.backward() çağrısı ile geriye doğru ilerletin. PyTorch, her bir parametreye göre kaybın gradyanlarını biriktirir.
# Gradyanlarımızı aldıktan sonra, parametreleri geriye doğru geçişte toplanan gradyanlara göre ayarlamak için optimizer.step() işlevini çağırırız.

# Tam Uygulama
# Optimizasyon kodumuz üzerinde döngü yapan train_loop ve modelin performansını test verilerimize göre değerlendiren test_loop tanımlıyoruz.
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# Kayıp fonksiyonunu ve optimize ediciyi başlatıp train_loop ve test_loop'a aktarıyoruz. Modelin gelişen performansını izlemek için epok sayısını artırmaktan çekinmeyin.
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
