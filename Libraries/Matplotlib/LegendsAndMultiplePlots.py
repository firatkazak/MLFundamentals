import matplotlib.pyplot as plt

stock_a = [100, 102, 99, 101, 101, 100, 102]
stock_b = [90, 95, 102, 104, 105, 103, 109]
stock_c = [110, 115, 100, 95, 100, 98, 95]
plt.plot(stock_a, label="Nike")
plt.plot(stock_b, label="Adidas")
plt.plot(stock_c, label="Puma")
plt.legend(loc="lower center")
picture_path = "C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Matplotlib/Data/Outputs"
file_name = "multiple_plot_with_legend.png"
plt.savefig(f"{picture_path}/{file_name}", dpi=300, bbox_inches='tight')
plt.close()