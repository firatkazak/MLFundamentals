import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb
from keras.callbacks import History

# IMDb veri setini yükle
max_features = 10000
maxlen = 100  # Her incelemeyi 100 kelimeye kadar kesiyoruz
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

# İncelemeleri pad_sequences ile aynı uzunluğa getir
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

# RNN modelini oluştur
model = Sequential()
model.add(Embedding(max_features, 128))  # Embedding katmanı
model.add(SimpleRNN(128))  # Basit RNN katmanı
model.add(Dense(1, activation='sigmoid'))  # Çıktı katmanı

# Modeli derle
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modelin eğitimini gerçekleştirmek için bir History nesnesi oluştur
history = History()
model.fit(X_train, y_train, batch_size=32, epochs=3, validation_split=0.2, callbacks=[history])

# Eğitim ve doğrulama kayıplarını ve doğruluklarını görselleştir
plt.figure(figsize=(12, 6))

# Kayıp grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Kayıp Grafiği')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()

# Doğruluk grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Doğruluk Grafiği')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()

plt.tight_layout()
plt.show()
