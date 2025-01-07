import matplotlib.pyplot as plt
import seaborn as sns

picture_path = "C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Seaborn/Data/Outputs"

crash_df = sns.load_dataset('car_crashes')  # car_crashes veri setini yükle
numeric_df = crash_df.select_dtypes(include=[float, int])  # Sadece sayısal sütunları seç
crash_mx = numeric_df.corr()  # Korelasyon matrisini hesapla
plt.figure(figsize=(8, 6))  # Grafik boyutunu ayarla
sns.set_context(context='paper', font_scale=1.4)  # Grafik bağlamını ayarla
sns.heatmap(crash_mx, annot=True, cmap='Blues')  # Korelasyon matrisini çiz

file_name = "heatmap1.png"
plt.savefig(f"{picture_path}/{file_name}", dpi=300, bbox_inches='tight')

flights = sns.load_dataset("flights")
flights = flights.pivot_table(index='month', columns='year', values='passengers', observed=False)
plt.figure(figsize=(8, 6))
sns.set_context(context='paper', font_scale=1.4)
sns.heatmap(flights, cmap='Blues', linecolor='white', linewidth=1)

file_name = "heatmap2.png"
plt.savefig(f"{picture_path}/{file_name}", dpi=300, bbox_inches='tight')

plt.close()
