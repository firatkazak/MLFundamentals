import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import DataLoader

# Datasets ve DataLoaders
# Veri örneklerini işleme kodu dağınık ve bakımı zor olabilir;
# ideal olarak daha iyi okunabilirlik ve modülerlik için veri kümesi kodumuzun model eğitim kodumuzdan ayrılmasını isteriz.
# PyTorch iki veri ilkeli sağlar: torch.utils.data.DataLoader ve torch.utils.data.Dataset.
# Bunlar, önceden yüklenmiş veri kümelerinin yanı sıra kendi verilerinizi de kullanmanıza olanak tanır.
# Dataset, örnekleri ve bunlara karşılık gelen etiketleri saklar ve DataLoader, örneklere kolay erişim sağlamak için Dataset'in etrafına bir iterable sarar.
# PyTorch alan kütüphaneleri, torch.utils.data.Dataset alt sınıfını oluşturan ve belirli verilere özgü işlevleri uygulayan bir dizi önceden yüklenmiş veri kümesi (FashionMNIST gibi) sağlar. Modelinizi prototiplemek ve kıyaslamak için kullanılabilirler.
# Onları aşağıdaki linklerde bulabilirsiniz: Görüntü Veri Kümeleri, Metin Veri Kümeleri ve Ses Veri Kümeleri
# Görüntü Veri Kümeleri: https://pytorch.org/vision/stable/datasets.html
# Metin Veri Kümeleri: https://pytorch.org/text/stable/datasets.html
# Ses Veri Kümeleri: https://pytorch.org/audio/stable/datasets.html

# Bir Veri Kümesi Yükleme: Burada TorchVision'dan Fashion-MNIST veri kümesinin nasıl yükleneceğine dair bir örnek verilmiştir.
# Fashion-MNIST, 60.000 eğitim örneği ve 10.000 test örneğinden oluşan Zalando'nun makale görüntülerinden oluşan bir veri kümesidir.
# Her örnek 28×28 gri tonlamalı bir görüntü ve 10 sınıftan biriyle ilişkili bir etiket içerir.
# FashionMNIST Veri Kümesini aşağıdaki parametrelerle yüklüyoruz:
# root, eğitim/test verilerinin depolandığı yoldur,
# train, eğitim veya test veri kümesini belirtir,
# download=True kökte mevcut değilse verileri internetten indirir.
# transform ve target_transform özellik ve etiket dönüşümlerini belirtir.

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

# Veri Kümesini İterleme ve Görselleştirme
# Veri kümelerini manuel olarak bir liste gibi indeksleyebiliriz: training_data[index].
# Eğitim verilerimizdeki bazı örnekleri görselleştirmek için matplotlib kullanıyoruz.

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3

for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")

plt.show()


# Dosyalarınız için Özel Veri Kümesi Oluşturma
# Özel bir Veri Kümesi sınıfı üç işlevi uygulamalıdır: __init__, __len__ ve __getitem__.
# FashionMNIST görüntüleri bir img_dir dizininde saklanır ve etiketleri bir CSV dosyası annotations_file'da ayrı olarak saklanır.


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# __init__: __init__ işlevi, Veri Kümesi nesnesi örneklenirken bir kez çalıştırılır. Görüntüleri, açıklama dosyasını ve her iki dönüşümü içeren dizini başlatırız.
# __len__: len__ fonksiyonu veri kümemizdeki örnek sayısını döndürür.
# __getitem__: getitem__ fonksiyonu, verilen idx indeksindeki veri kümesinden bir örnek yükler ve döndürür. İndekse bağlı olarak, görüntünün diskteki konumunu belirler, read_image kullanarak bunu bir tensöre dönüştürür, self.img_labels içindeki csv verilerinden karşılık gelen etiketi alır, bunlar üzerindeki dönüştürme işlevlerini çağırır (varsa) ve tensör görüntüsünü ve karşılık gelen etiketi bir tuple olarak döndürür.

# DataLoaders ile verilerinizi eğitim için hazırlama
# Veri Kümesi, veri kümemizin özelliklerini ve etiketlerini her seferinde bir örnek olarak alır.
# Bir modeli eğitirken, genellikle örnekleri “mini gruplar(mini_batchs)” halinde iletmek, modelin aşırı uyumunu azaltmak için her epokta verileri yeniden karıştırmak ve veri alımını hızlandırmak için Python'un çoklu işlemesini kullanmak isteriz.
# DataLoader, bu karmaşıklığı bizim için kolay bir API ile soyutlayan bir yinelenebilirdir.

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# DataLoader aracılığıyla itere etme
# Bu veri kümesini DataLoader'a yükledik ve gerektiğinde veri kümesi üzerinde yineleme yapabiliriz.
# Aşağıdaki her yineleme train_features ve train_labels (sırasıyla batch_size=64 özellik ve etiket içeren) yığınlarını döndürür.
# shuffle=True olarak belirttiğimiz için, tüm gruplar üzerinde yineleme yaptıktan sonra veriler karıştırılır.

# Görüntüyü ve etiketi gösterme.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
