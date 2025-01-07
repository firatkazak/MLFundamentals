import plotly.express as px
import seaborn as sns

# Seaborn verilerini kullanarak bir ısı haritası oluşturma;
flights = sns.load_dataset("flights")

# Bin'leri nbinsx ve nbinsy ile ayarlayabilirsiniz;
fig1 = px.density_heatmap(flights, x='year', y='month', z='passengers', color_continuous_scale="Viridis")

# Histogram ekleyebilirsiniz;
fig2 = px.density_heatmap(flights, x='year', y='month', z='passengers', marginal_x="histogram", marginal_y="histogram")

fig1.show() # 1. YOL
fig2.show() # 2. YOL
