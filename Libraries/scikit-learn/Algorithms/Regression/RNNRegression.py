import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

# Sinüs dalgası oluştur
x = np.linspace(0, 100, 500)
y = np.sin(x)


# Veri setini oluştur (örnekleme)
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step)])
        Y.append(data[i + time_step])
    return np.array(X), np.array(Y)


# Zaman adımı
time_step = 10
X, Y = create_dataset(y, time_step)

# Veriyi yeniden şekillendir
X = X.reshape(X.shape[0], X.shape[1], 1)  # [örnek sayısı, zaman adımı, özellik sayısı]

# RNN modelini oluştur
model = Sequential()
model.add(SimpleRNN(50, input_shape=(X.shape[1], 1)))  # RNN katmanı
model.add(Dense(1))  # Çıktı katmanı

# Modeli derle
model.compile(loss='mean_squared_error', optimizer='adam')

# Modeli eğit
model.fit(X, Y, epochs=100, batch_size=32)

# Tahmin yap
Y_pred = model.predict(X)

# Sonuçları görselleştir
plt.plot(y[time_step + 1:], label='Gerçek')
plt.plot(Y_pred, label='Tahmin')
plt.title('Sinüs Dalga Regresyon')
plt.legend()
plt.show()
