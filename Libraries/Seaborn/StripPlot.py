import matplotlib.pyplot as plt
import seaborn as sns

sns.get_dataset_names()
tips_df = sns.load_dataset('tips')
plt.figure(figsize=(8, 5))
sns.stripplot(x='day', y='total_bill', data=tips_df, jitter=True, hue='sex', dodge=True)

picture_path = "C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Seaborn/Data/Outputs"
file_name = "strip_plot.png"
plt.savefig(f"{picture_path}/{file_name}", dpi=300, bbox_inches='tight')
plt.close()