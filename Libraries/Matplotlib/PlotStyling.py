import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")
votes = [10, 2, 5, 16, 22]
people = ["Ali", "Barış", "Can", "Deniz", "Efe"]
plt.pie(votes, labels=None)
plt.legend(labels=people)

picture_path = "C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Matplotlib/Data/Outputs"
file_name = "plot_styling.png"
plt.savefig(f"{picture_path}/{file_name}", dpi=300, bbox_inches='tight')
plt.close()
