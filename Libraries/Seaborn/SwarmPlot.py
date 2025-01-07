import matplotlib.pyplot as plt
import seaborn as sns

sns.get_dataset_names()  # Mevcut veri setlerinin isimlerini yazdır
tips_df = sns.load_dataset('tips')  # "tips" veri setini yükle
plt.figure(figsize=(8, 6))  # Grafik boyutunu ayarla
sns.set_style('dark')  # Stil ayarlama
sns.set_context('talk')  # Bağlam ayarlama
sns.swarmplot(x='day', y='total_bill', data=tips_df, hue="day", palette="seismic")  # Verileri çiz

picture_path = "C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Seaborn/Data/Outputs"
file_name = "swarm_plot.png"
plt.savefig(f"{picture_path}/{file_name}", dpi=300, bbox_inches='tight')
plt.close()