import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# Başlangıç verileri
head_tails = [0, 0]
labels = ["Heads", "Tails"]
colors = ["red", "blue"]

# Şekil ve eksen oluşturma
fig, ax = plt.subplots()
bars = ax.bar(labels, head_tails, color=colors)

# Y ekseni sınırını belirleyelim
ax.set_ylim(0, 100000)


# Animasyon fonksiyonu
def update(frame):
    global head_tails
    # Rastgele kafa veya tura artır
    head_tails[random.randint(0, 1)] += 1
    # Barların yüksekliklerini güncelle
    for bar, height in zip(bars, head_tails):
        bar.set_height(height)
    return bars


# Animasyon ayarları
anim = FuncAnimation(fig, update, frames=range(100000), interval=1, blit=True)

# Animasyonu gösterme
plt.show()
