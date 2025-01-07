import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# CIFAR-10 veri kümesini yükleyin
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Resimleri 0-1 aralığına yeniden ölçekleyin
train_images = train_images / 255.0
test_images = test_images / 255.0

# Tensörleri NumPy dizilerine çevir
train_images_np = train_images.copy()  # Tensörleri kopyalayıp NumPy dizilerine dönüştürüyoruz
train_labels_np = train_labels.copy()

# train_test_split ile veri setini küçültelim (Eğitim veri setinin %90'ını çıkartıyoruz)
train_images, _, train_labels, _ = train_test_split(train_images_np, train_labels_np, test_size=0.9, random_state=42)

# Önceden eğitilmiş MobileNetV2 modelini yükleyin (include_top=False ile sınıflandırma katmanları hariç)
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Modelin eğitilmesini istemediğimiz katmanları dondur
base_model.trainable = False

# Yeni model oluşturun
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(units=128, activation='relu'),
    layers.Dense(units=10, activation='softmax')  # 10 sınıf olduğu için softmax kullanıyoruz
])

# Modeli derleyin
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitin
model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=3, batch_size=32)

# Modeli kaydedin
# Modeli belirttiğin klasöre kaydet
model.save(r'C:\Users\firat\source\repos\FiratMLDersleri\Kaydedilenler\light_transfer_learning_model.keras')
