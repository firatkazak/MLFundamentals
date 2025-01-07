import matplotlib.pyplot as plt
import seaborn as sns

picture_path = "C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Seaborn/Data/Outputs"

# Veri Seti;
tips_df = sns.load_dataset("tips")

# 1. Veri;
tips_fg = sns.FacetGrid(tips_df, col='time', row='smoker')
tips_fg.map(plt.hist, "total_bill", bins=8)
tips_fg.map(plt.scatter, "total_bill", "tip")

file_name = "faced_grid1.png"
plt.savefig(f"{picture_path}/{file_name}", dpi=300, bbox_inches='tight')

# 2. Veri
tips_fg = sns.FacetGrid(tips_df, col='time', hue='smoker', height=4, aspect=1.3, col_order=['Dinner', 'Lunch'], palette='Set1')
tips_fg.map(plt.scatter, "total_bill", "tip", edgecolor='w')
kws = dict(s=50, linewidth=.5, edgecolor="w")

file_name = "faced_grid2.png"
plt.savefig(f"{picture_path}/{file_name}", dpi=300, bbox_inches='tight')

# 3. Veri;
tips_fg = sns.FacetGrid(tips_df, col='sex', hue='smoker', height=4, aspect=1.3, hue_order=['Yes', 'No'], hue_kws=dict(marker=['^', 'v']))
tips_fg.map(plt.scatter, "total_bill", "tip", **kws)
tips_fg.add_legend()
att_df = sns.load_dataset("attention")

file_name = "faced_grid3.png"
plt.savefig(f"{picture_path}/{file_name}", dpi=300, bbox_inches='tight')

# 4. Veri;
att_fg = sns.FacetGrid(att_df, col='subject', col_wrap=5, height=1.5)
att_fg.map(plt.plot, 'solutions', 'score', marker='.')

file_name = "faced_grid4.png"
plt.savefig(f"{picture_path}/{file_name}", dpi=300, bbox_inches='tight')

plt.close()
