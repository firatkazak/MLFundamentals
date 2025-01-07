import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# MNIST veri kümesini yükle
(X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5  # Normalize et
X_train = np.expand_dims(X_train, axis=-1)  # (60000, 28, 28) => (60000, 28, 28, 1)

# GAN için hiperparametreler
latent_dim = 50  # Rastgele gürültü boyutu
epochs = 1000  # Epoch sayısını düşürdüm
batch_size = 64  # Batch boyutunu düşürdüm


# Generatör modelini tanımla
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', ),
        layers.Dense(256, activation='relu'),
        layers.Dense(28 * 28 * 1, activation='tanh'),
        layers.Reshape((28, 28, 1))
    ])
    return model


# Ayırıcı modelini tanımla
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


# Generatör ve Ayırıcıyı oluştur
generator = build_generator()
discriminator = build_discriminator()

# Ayırıcıyı derle
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# GAN modelini tanımla
discriminator.trainable = False
gan_input = layers.Input(shape=(latent_dim,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)
gan = tf.keras.Model(gan_input, gan_output)

# GAN modelini derle
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Eğitim döngüsü
for epoch in range(epochs):
    # Rastgele gerçek görüntüler
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_images = X_train[idx]

    # Rastgele gürültü oluştur
    noise_input = np.random.normal(0, 1, (batch_size, latent_dim))  # noise değişkenini değiştirdim
    fake_images = generator.predict(noise_input)

    # Ayırıcıyı eğit
    d_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Generatörü eğit
    noise_gen = np.random.normal(0, 1, (batch_size, latent_dim))  # noise değişkenini değiştirdim
    g_loss = gan.train_on_batch(noise_gen, np.ones((batch_size, 1)))

    # Eğitim ilerlemesini yazdır
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Discriminator Loss: {d_loss[0]}, Generator Loss: {g_loss}")


# Sonuçları görselleştir
def plot_generated_images(generator_model, examples=10, dim=(1, 10), figsize=(10, 1)):  # generator yerine generator_model kullandım
    noise_samples = np.random.normal(0, 1, (examples, latent_dim))  # noise değişkenini değiştirdim
    generated_images = generator_model.predict(noise_samples)
    generated_images = 0.5 * generated_images + 0.5  # Normalize et
    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.show()


# Görüntüleri göster
plot_generated_images(generator)
