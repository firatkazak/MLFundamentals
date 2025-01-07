import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.datasets import load_iris

# Iris veri setini yükle
data = load_iris()
X = data.data  # Özellikler

# Basit bir autoencoder oluştur
input_data = Input(shape=(4,))
encoded = Dense(2, activation='relu')(input_data)  # Boyut indirgeme
decoded = Dense(4, activation='sigmoid')(encoded)  # Yeniden yapılandırma

autoencoder = Model(input_data, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Autoencoder'ı eğit
autoencoder.fit(X, X, epochs=50, batch_size=16, shuffle=True)

# Encoder kısmını çıkar
encoder = Model(input_data, encoded)
X_encoded = encoder.predict(X)

# Veriyi görselleştir
plt.scatter(X_encoded[:, 0], X_encoded[:, 1])
plt.title('Autoencoder ile Boyut İndirgeme')
plt.xlabel('Kodlanmış 1. Bileşen')
plt.ylabel('Kodlanmış 2. Bileşen')
plt.show()

# Autoencoders
# Alegori: Autoencoder, bir çevirmenin hikayesidir.
# Çevirmen, her iki dilden de en önemli kelimeleri seçerek,
# anlamı kaybetmeden özlü bir çeviri yapar.
