import matplotlib.pyplot as plt
import seaborn as sns

picture_path = "C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Seaborn/Data/Outputs"

tips_df = sns.load_dataset('tips')

# 1. Veri;
sns.set_context(context='paper', font_scale=1.4)
sns.lmplot(x='total_bill',
           y='tip',
           hue='sex',
           data=tips_df,
           markers=['o', '^'],
           scatter_kws={'s': 100, 'linewidths': 0.5, 'edgecolor': 'w'}
           )
file_name = "regression_plot1.png"
plt.savefig(f"{picture_path}/{file_name}", dpi=300, bbox_inches='tight')
# 2. Veri;
sns.set_context(context='poster', font_scale=1.4)
sns.lmplot(x='total_bill',
           y='tip',
           data=tips_df,
           col='day',
           hue='sex',
           height=8,
           aspect=0.6
           )

file_name = "regression_plot2.png"
plt.savefig(f"{picture_path}/{file_name}", dpi=300, bbox_inches='tight')

plt.close()
