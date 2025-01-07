import numpy as np
import matplotlib.pyplot as plt

ages = np.random.normal(loc=20, scale=1.5, size=1000)
plt.hist(x=ages,
         bins=20,  # kutu sayısı
         cumulative=True  # kümülatif olarak ayarlar
         )

picture_path = "C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Matplotlib/Data/Outputs"
file_name = "histogram.png"
plt.savefig(f"{picture_path}/{file_name}", dpi=300, bbox_inches='tight')
plt.close()