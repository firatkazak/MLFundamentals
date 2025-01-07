import matplotlib.pyplot as plt
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.datasets import make_blobs

# Örnek veri seti oluşturun
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# K-Medoids algoritmasını uygulayın
# İlk medoidlerin indekslerini rastgele seçin
initial_medoids = [0, 50, 100, 150]  # Örneğin, ilk dört veriyi medoid olarak seçiyoruz
kmedoids_instance = kmedoids(X.tolist(), initial_medoids)
kmedoids_instance.process()

# Tahmin edilen küme etiketleri
clusters = kmedoids_instance.get_clusters()

# Sonuçları görselleştirin
for cluster in clusters:
    plt.scatter(X[cluster, 0], X[cluster, 1])

# Medoidleri işaretleyin
medoids = kmedoids_instance.get_medoids()
plt.scatter(X[medoids, 0], X[medoids, 1], c='red', s=200, marker='X', label='Medoidler')

plt.title('K-Medoids Kümeleme')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.legend()
plt.show()

# K-Medoids
# Veri Seti Oluşturma: make_blobs fonksiyonu kullanarak rastgele bir veri seti oluşturuyoruz. Bu veri seti 4 farklı merkez etrafında kümelenmiş 300 örnek içeriyor.
# K-Medoids Uygulaması: kmedoids sınıfını kullanarak K-Medoids algoritmasını uyguluyoruz. initial_medoids ile ilk medoidlerin indekslerini belirtiyoruz.
# Tahmin ve Görselleştirme: Model eğitildikten sonra, tahmin edilen küme etiketlerini alıyoruz ve sonuçları matplotlib ile görselleştiriyoruz.
# Medoidler kırmızı "X" ile gösteriliyor.
