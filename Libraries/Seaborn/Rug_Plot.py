import matplotlib.pyplot as plt
import seaborn as sns

sns.get_dataset_names()
tips_df = sns.load_dataset('tips')
sns.rugplot(tips_df["tip"])

picture_path = "C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Seaborn/Data/Outputs"
file_name = "rug_plot.png"
plt.savefig(f"{picture_path}/{file_name}", dpi=300, bbox_inches='tight')
plt.close()
