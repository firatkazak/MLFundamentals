import matplotlib.pyplot as plt
import seaborn as sns

iris = sns.load_dataset("iris")
flights = sns.load_dataset("flights")
flights = flights.pivot_table(index='month', columns='year', values='passengers', observed=False)
plt.figure(figsize=(8, 6))
sns.set_context(context='paper', font_scale=1.4)
sns.clustermap(flights, cmap="Blues", standard_scale=1)

picture_path = "C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Seaborn/Data/Outputs"
file_name = "cluster_map.png"
plt.savefig(f"{picture_path}/{file_name}", dpi=300, bbox_inches='tight')
plt.close()
