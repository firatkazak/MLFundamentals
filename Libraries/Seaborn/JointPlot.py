import matplotlib.pyplot as plt
import seaborn as sns

sns.get_dataset_names()
crash_df = sns.load_dataset('car_crashes')
sns.jointplot(x='speeding', y='alcohol', data=crash_df, kind='reg')

picture_path = "C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Seaborn/Data/Outputs"
file_name = "joint_plot.png"
plt.savefig(f"{picture_path}/{file_name}", dpi=300, bbox_inches='tight')
plt.close()