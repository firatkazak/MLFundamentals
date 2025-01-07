import numpy as np
import tensorflow as tf


# Basit bir Attention mekanizması fonksiyonu
def scaled_dot_product_attention(query, key, value, mask=None):
    """Sorgu, anahtar ve değer matrisiyle Scaled Dot-Product Attention hesaplar."""

    # Sorgu ve anahtarların transpozuyla çarpım
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # Ölçekleme
    dk = tf.cast(tf.shape(key)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Opsiyonel maskeleme
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  # Büyük negatif sayılar

    # Softmax uygulayarak ağırlıkları bul
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # Ağırlıklarla değer matrisini çarp
    output = tf.matmul(attention_weights, value)

    return output, attention_weights


# Sorgu, Anahtar ve Değer matrisleri için örnek veriler
np.random.seed(0)
query = tf.constant(np.random.random((1, 3, 5)), dtype=tf.float32)  # [batch_size, seq_len, d_k]
key = tf.constant(np.random.random((1, 3, 5)), dtype=tf.float32)  # [batch_size, seq_len, d_k]
value = tf.constant(np.random.random((1, 3, 5)), dtype=tf.float32)  # [batch_size, seq_len, d_v]

# Maske oluşturma (örneğin, 1. ve 2. öğelerin dikkate alınmamasını istiyoruz)
mask = tf.constant([[0, 1, 1],  # 1. öğeye dikkat edilmesin
                    [1, 0, 1],  # 2. öğeye dikkat edilmesin
                    [1, 1, 1]], dtype=tf.float32)  # 3. öğeye dikkat edilsin

# Attention hesaplama
output, attention_weights = scaled_dot_product_attention(query, key, value, mask)

print("Output (Dikkat sonuçları):", output)
print("Attention Weights (Dikkat Ağırlıkları):", attention_weights)

# AÇIKLAMA;
# Attention mekanizması, bir modelin belirli girdilere daha fazla dikkat etmesine olanak tanıyarak, önemli bilgilere odaklanmasını sağlar.

# 1. Temel Bileşenler;

# Query (Sorgu): Modelin neyi aradığını temsil eder. Diğer öğelerle (anahtarlar) karşılaştırılır.

# Key (Anahtar): Girdi öğelerinin temsilidir. Her bir anahtar, ilgili değerin ne kadar önemli olduğunu belirlemek için sorguyla karşılaştırılır.

# Value (Değer): Sorguya yanıt olarak döndürülen bilgilerdir. Dikkat mekanizması, hangi değerlere (output) ne kadar dikkat edilmesi gerektiğini belirler.

# 2. Dikkat Hesaplama Süreci;

# Sorgu ve Anahtar Çarpımı: Sorgu ve anahtar matrisleri arasındaki çarpım, hangi anahtarların sorguya ne kadar uygun olduğunu ölçer.
# Bu, scaled_dot_product_attention fonksiyonundaki matmul_qk değişkeniyle yapılır.

# Ölçekleme: Sonuç, anahtarların boyutuna göre ölçeklenir. Bu, büyük sayıların softmax uygulandığında aşırı ağırlıklandırma yapmasını önler.
# Bu işlem scaled_attention_logits'te yapılır.

# Maskeleme (Opsiyonel): Eğer belirli öğelere dikkat etmemek isteniyorsa, maskeler kullanılarak bu öğelerin dikkati düşürülür.

# Softmax: Ölçeklenmiş çarpım sonuçları, dikkat ağırlıklarını bulmak için softmax fonksiyonuna tabi tutulur. Bu, toplamı 1 olacak şekilde normalleştirir.
#
# Değerlerle Çarpma: Elde edilen dikkat ağırlıkları, değer matrisine uygulanır.
# Bu, hangi değerlerin daha fazla etkili olacağını belirler. Sonuç, modelin hangi bilgilere dikkat ettiğini gösterir.

# 3. Sonuç
# Output: Bu, dikkat mekanizmasının sonunda elde edilen çıktıdır. Dikkat ağırlıklarıyla çarpılan değerlerden oluşur.
# Modelin, hangi girişlere daha fazla dikkat ettiğini ve bunun sonucunda hangi bilgilerin üretildiğini gösterir.

# Bu mekanizma, özellikle dil modelleri ve makine çevirisi gibi görevlerde önemli bir rol oynar.
# Çünkü her bir kelime veya öğe, bağlamına bağlı olarak değişik derecelerde önem taşır.
# Attention sayesinde, model belirli kelimelere veya öğelere daha fazla odaklanarak daha anlamlı çıktılar üretebilir.
