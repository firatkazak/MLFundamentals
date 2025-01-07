import matplotlib.pyplot as plt
import seaborn as sns

sns.get_dataset_names()
tips_df = sns.load_dataset('tips')

# Tek bir figür oluştur ve 1 satır, 2 sütunluk bir grid oluştur
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# İlk subplot'a boxplot çizdir
sns.boxplot(x='day', y='total_bill', data=tips_df, hue='sex', ax=axes[0])
axes[0].legend(loc=0)

# İkinci subplot'a farklı bir grafik çizdir (örneğin, histogram)
sns.histplot(data=tips_df, x='total_bill', kde=True, ax=axes[1])

picture_path = "C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Seaborn/Data/Outputs"
file_name = "box_plot.png"
plt.savefig(f"{picture_path}/{file_name}", dpi=300, bbox_inches='tight')
plt.close()