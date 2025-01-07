import numpy as np
import matplotlib.pyplot as plt

picture_path = "C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Matplotlib/Data/Outputs"

# 1. Örnek
heights = np.random.normal(loc=172, scale=8, size=300)
plt.boxplot(heights)
file_name = "boxplot1.png"
plt.savefig(f"{picture_path}/{file_name}", dpi=300, bbox_inches='tight')
# 2. Örnek
first = np.linspace(start=0, stop=10, num=25)
second = np.linspace(start=10, stop=200, num=25)
third = np.linspace(start=200, stop=210, num=25)
fourth = np.linspace(start=210, stop=230, num=25)

data = np.concatenate((first, second, third, fourth))
plt.boxplot(data)
file_name = "boxplot2.png"
plt.savefig(f"{picture_path}/{file_name}", dpi=300, bbox_inches='tight')
plt.close()
