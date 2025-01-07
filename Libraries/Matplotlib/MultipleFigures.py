import numpy as np
import matplotlib.pyplot as plt

x1, y1 = np.random.random(100), np.random.random(100)
x2, y2 = np.arange(100), np.random.random(100)
plt.figure(1)
plt.scatter(x1, y1)
plt.figure(2)
plt.plot(x2, y2)

picture_path = "C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Matplotlib/Data/Outputs"
file_name = "multiple_figures.png"
plt.savefig(f"{picture_path}/{file_name}", dpi=300, bbox_inches='tight')
plt.close()
