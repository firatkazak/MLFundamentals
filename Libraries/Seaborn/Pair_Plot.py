import matplotlib.pyplot as plt
import seaborn as sns

sns.get_dataset_names()
tips_df = sns.load_dataset('tips')
sns.pairplot(tips_df, hue="sex", palette="Purples")

picture_path = "C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Seaborn/Data/Outputs"
file_name = "pair_plot.png"
plt.savefig(f"{picture_path}/{file_name}", dpi=300, bbox_inches='tight')
plt.close()
