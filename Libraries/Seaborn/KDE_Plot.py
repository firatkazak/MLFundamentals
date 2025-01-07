import matplotlib.pyplot as plt
import seaborn as sns

sns.get_dataset_names()
crash_df = sns.load_dataset('car_crashes')
sns.kdeplot(crash_df['alcohol'])

picture_path = "C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Seaborn/Data/Outputs"
file_name = "kde_plot.png"
plt.savefig(f"{picture_path}/{file_name}", dpi=300, bbox_inches='tight')
plt.close()
