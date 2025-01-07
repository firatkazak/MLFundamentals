import matplotlib.pyplot as plt

x = ["C++", "C#", "Python", "Java", "Go"]
y = [20, 50, 140, 1, 45]
plt.bar(x=x,
        height=y,
        color="blue",
        align="edge",
        width=0.5,
        edgecolor="yellow",
        lw=6
        )

picture_path = "C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Matplotlib/Data/Outputs"
file_name = "bar_plot.png"
plt.savefig(f"{picture_path}/{file_name}", dpi=300, bbox_inches='tight')
plt.close()