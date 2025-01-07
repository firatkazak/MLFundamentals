import matplotlib.pyplot as plt
import seaborn as sns

sns.get_dataset_names()
crash_df = sns.load_dataset('car_crashes')
sns.set_style("ticks")
plt.figure(figsize=(8, 4))
sns.set_context(context="talk", font_scale=1.4)
sns.jointplot(x="speeding", y="alcohol", data=crash_df)
sns.despine(left=False, bottom=False)

picture_path = "C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Seaborn/Data/Outputs"
file_name = "styling.png"
plt.savefig(f"{picture_path}/{file_name}", dpi=300, bbox_inches='tight')
plt.close()
