import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.get_dataset_names()
tips_df = sns.load_dataset('tips')
sns.barplot(x='sex', y='total_bill', color="yellow", data=tips_df, estimator=np.median, hue="sex", palette={'Male': 'red', 'Female': 'yellow'}, dodge=False)

picture_path = "C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Seaborn/Data/Outputs"
file_name = "bar_plot.png"
plt.savefig(f"{picture_path}/{file_name}", dpi=300, bbox_inches='tight')
plt.close()
