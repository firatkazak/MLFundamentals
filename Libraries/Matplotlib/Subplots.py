import numpy as np
import matplotlib.pyplot as plt

x = np.arange(100)

fig, axs = plt.subplots(nrows=2, ncols=2)

axs[0, 0].plot(x, np.sign(x))
axs[0, 0].set_title("Sine Wave")

axs[0, 1].plot(x, np.cos(x))
axs[0, 1].set_title("Cosine Wave")

axs[1, 0].plot(x, np.random.random(100))
axs[1, 0].set_title("Random Function")

axs[1, 1].plot(x, np.log(x))
axs[1, 1].set_title("Log Wave")

fig.suptitle("4 Grafik")

picture_path = "C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Matplotlib/Data/Outputs"
file_name = "subpilot.png"
plt.savefig(f"{picture_path}/{file_name}", dpi=300, bbox_inches='tight')
plt.close()
