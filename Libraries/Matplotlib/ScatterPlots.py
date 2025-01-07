import numpy as np
import matplotlib.pyplot as plt

# Veri oluşturma
X_data = np.random.random(50) * 100
y_data = np.random.random(50) * 100

# Grafik çizimi
plt.scatter(x=X_data,  # x verisi
            y=y_data,  # y verisi
            c="red",  # renk
            marker="*",  # işaret tipi
            s=150,  # boyut
            alpha=0.3  # saydamlık
            )

# Resmi kaydetme
picture_path = "C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Matplotlib/Data/Outputs"
file_name = "scatter_plot.png"  # Kaydedilecek dosyanın adı
plt.savefig(f"{picture_path}/{file_name}", dpi=300, bbox_inches='tight')  # DPI ve dış kenarları optimize et

# Kaydedildikten sonra görüntülemeye gerek yoksa `plt.show()` kullanılmaz.
plt.close()  # Bellekte yer tutmaması için grafiği kapat
