import matplotlib.pyplot as plt
import seaborn as sns

tips_df = sns.load_dataset("tips")
iris = sns.load_dataset("iris")
iris_g = sns.PairGrid(iris, hue="species", x_vars=["sepal_length", "sepal_width"], y_vars=["petal_length", "petal_width"])
iris_g.map(plt.scatter)
iris_g.add_legend()

picture_path = "C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Seaborn/Data/Outputs"
file_name = "pair_grid.png"
plt.savefig(f"{picture_path}/{file_name}", dpi=300, bbox_inches='tight')
plt.close()
