import matplotlib.pyplot as plt

years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
income = [55, 56, 62, 61, 72, 72, 73, 75]
income_ticks = list(range(50, 81, 2))
plt.plot(years, income)
plt.title(label="Income of Fırat (in TL)", fontsize=30, fontname="Arial")
plt.xlabel("Year")
plt.ylabel("Income in TL")
plt.yticks(income_ticks, [f"{x}k TL" for x in income_ticks])

picture_path = "C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Matplotlib/Data/Outputs"
file_name = "plot_customization.png"
plt.savefig(f"{picture_path}/{file_name}", dpi=300, bbox_inches='tight')
plt.close()
