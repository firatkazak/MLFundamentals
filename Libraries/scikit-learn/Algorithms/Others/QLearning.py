import numpy as np
import gym

# Çevreyi oluşturun
env = gym.make('CartPole-v1')

# Q-öğrenme için parametreler
n_episodes = 1000  # Toplam epizod sayısı
if isinstance(env.action_space, gym.spaces.Discrete):
    n_actions = env.action_space.n  # Aksiyon sayısı
else:
    raise ValueError("Aksiyon alanı Discrete değil.")

# Durum sayısını belirlemek için CartPole için durumları ayrıştırmamız gerekiyor
n_states = 10  # Bu örnekte 10 ayrı duruma ayrıştırıyoruz

# Q tablosunu başlatın
Q = np.zeros((n_states, n_actions))

# Hiperparametreler
alpha = 0.1  # Öğrenme oranı
gamma = 0.99  # İndirim oranı
epsilon = 1.0  # Keşif oranı
epsilon_decay = 0.995  # Epsilon'un azalması


def discretize(cartpole_state):
    # Durumu ayrıştır (discretization)
    return int(np.digitize(cartpole_state[2], np.linspace(-2.0, 2.0, n_states)))


# Q-öğrenme döngüsü
for episode in range(n_episodes):
    state = env.reset()  # State burada 4 elemanlı bir dizi olmalı

    # Eğer state bir tuple ise, ilk elemanı al
    if isinstance(state, tuple):
        state = state[0]  # Tuple'dan durumu al

    print(f"Initial state: {state}")  # Başlangıç durumunu yazdır
    done = False

    while not done:
        state_index = discretize(state)  # state_index burada tanımlanmalı

        # Epsilon-greedy stratejisi ile aksiyon seç
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Rastgele aksiyon
        else:
            action = np.argmax(Q[state_index])  # En iyi aksiyonu seç

        # Aksiyonu uygula ve geri bildirim al
        next_state, reward, done, truncated, _ = env.step(action)  # Dördüncü değer (truncated) eklendi

        # Durumu güncelle ve ayrıştır
        if isinstance(next_state, tuple):
            next_state = next_state[0]  # Tuple'dan durumu al

        next_state_index = discretize(next_state)

        # Q değerini güncelle
        Q[state_index, action] += alpha * (reward + gamma * np.max(Q[next_state_index]) - Q[state_index, action])

        # Durumu güncelle
        state = next_state

    # Epsilon'u güncelle
    epsilon *= epsilon_decay

# Çevreyi kapat
env.close()

# Q-Learning:
# Q-Learning, pekiştirmeli öğrenmenin temel yöntemlerinden biridir ve makine öğrenmesinin derinliklerine inmenizi sağlar.
# Temel ilkeleri kavrayarak, daha karmaşık pekiştirmeli öğrenme algoritmalarını anlamak için sağlam bir temel oluşturursunuz.
# Her aşamada deneyim kazanarak, bu kavramları daha iyi anlayabilir ve uygulamalarınızda daha etkili sonuçlar elde edebilirsiniz.

# 1. Temel Kavramlar
# Markov Karar Süreçleri (MDP): Q-Learning, karar verme problemlerini çözmek için kullanılan bir yöntemdir.
# Markov Karar Süreçleri, belirli bir durumdan başlayarak gelecekteki durumları ve ödülleri tahmin etmek için kullanılır.
#
# Durum (State): Ortamın mevcut durumu. Örneğin, CartPole'da, kutunun konumu ve açısı gibi faktörler durumları belirler.
#
# Aksiyon (Action): Algoritmanın belirli bir durumda alabileceği eylemler. Örneğin, CartPole'da kutuyu sola veya sağa hareket ettirmek gibi.
#
# Ödül (Reward): Bir aksiyonun sonucunda alınan geri bildirim. Q-Learning'de, ödül, ajanınızın ortamda ne kadar başarılı olduğunu belirler.
#
# 2. Q-Tablosu
# Q-Değerleri: Her bir durum ve aksiyon çifti için beklenen ödülü temsil eden değerlerdir.
# Q-Tablosu, bu değerleri depolar. Q-Değerlerini güncelleyerek, ajanınızın öğrenme sürecini yönlendirirsiniz.
# 3. Öğrenme Süreci
# Epsilon-Greedy Stratejisi: Keşif (exploration) ve istismar (exploitation) arasında denge kurar.
# Başlangıçta yüksek bir keşif oranı ile rastgele eylemler seçerken, zamanla daha iyi bilgilere sahip oldukça en iyi eylemi seçme olasılığı artar.
#
# Q-Değer Güncellemesi: Q-Değerlerini güncelleyerek, ajanınızın geçmiş deneyimlerinden öğrenmesini sağlarsınız.

# 4. Pratik Uygulamalar
# Öğrenme Oranı (
# 𝛼
# α) ve İndirim Oranı (
# 𝛾
# γ): Bu iki parametre, öğrenme sürecinizin nasıl gideceğini belirler. Farklı ayarlarla deneyler yaparak hangi ayarların en iyi sonucu verdiğini gözlemleyebilirsiniz.
#
# Daha Karmaşık Ortamlar: CartPole basit bir örnektir. Q-Learning'i daha karmaşık ortamlarla deneyerek, daha fazla deneyim kazanabilirsiniz.

# 5. Gelişmiş Yöntemler
# Derin Q-Ağları (DQN): Q-Learning’in derin öğrenme ile birleştirilmiş hali, daha karmaşık durumları öğrenmek için kullanılır.
#
# Double Q-Learning: Aşırı tahminleri önlemek için iki ayrı Q-Tablosu kullanır.
#
# Prioritized Experience Replay: Deneyimlerin önceliklendirilmesi ile öğrenme sürecini iyileştirir.
