import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Veri seti
input_texts = ['merhaba', 'nasılsın?', 'güzel hava.']
target_texts = ['hello', 'how are you?', 'nice weather.']

# Tokenizasyon
tokenizer_inputs = keras.preprocessing.text.Tokenizer()
tokenizer_inputs.fit_on_texts(input_texts)
tokenizer_targets = keras.preprocessing.text.Tokenizer()
tokenizer_targets.fit_on_texts(target_texts)

input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)
target_sequences = tokenizer_targets.texts_to_sequences(target_texts)

# Girdi ve hedef dizilerini pad etme
max_input_length = max(len(seq) for seq in input_sequences)
max_target_length = max(len(seq) for seq in target_sequences)

input_sequences = keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_input_length)
target_sequences = keras.preprocessing.sequence.pad_sequences(target_sequences, maxlen=max_target_length)

# Encoder
encoder_inputs = layers.Input(shape=(None,))
encoder_embedding = layers.Embedding(input_dim=len(tokenizer_inputs.word_index) + 1, output_dim=64)(encoder_inputs)
encoder_lstm = layers.LSTM(64, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = layers.Input(shape=(None,))
decoder_embedding = layers.Embedding(input_dim=len(tokenizer_targets.word_index) + 1, output_dim=64)(decoder_inputs)
decoder_lstm = layers.LSTM(64, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = layers.Dense(len(tokenizer_targets.word_index) + 1, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Modeli oluşturma
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Modeli derleme
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Hedef dizisini uygun hale getirme
target_sequences = np.expand_dims(target_sequences, -1)  # (örnek sayısı, hedef uzunluğu, 1)

# Modeli eğitme
model.fit([input_sequences, target_sequences], target_sequences, epochs=100)

# Örnek tahmin
# İlk giriş cümlesinin LSTM çıkışını tahmin etme
input_sample = np.array([input_sequences[0]])
output_sample = model.predict([input_sample, np.zeros((1, max_target_length))])
predicted_sequence = np.argmax(output_sample, axis=-1)
print("Tahmin Edilen Çıktı:", tokenizer_targets.sequences_to_texts(predicted_sequence))

# AÇIKLAMA:
# İki Giriş: Model, hem girdi (encoder) hem de hedef (decoder) dizilerini alır.
# Model Çıkışları: Eğitimden sonra tahminler yapılarak çıktı alınabilir.
# Giriş ve Hedef Dizileri: Hedef dizileri, modelin eğitimi sırasında bir adım ileriye kaydırılır,
# böylece modelin çıktısı her zaman bir önceki çıktıya bağlı olur.

# Aldığım "hello" çıktısı, Diger modelinin verdiği bir tahmin.
# Modelin amacı, girdi olarak aldığı Türkçe bir cümleyi (örneğin, "merhaba") İngilizceye (örneğin, "hello") çevirmektir.
