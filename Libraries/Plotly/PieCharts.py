import plotly.graph_objects as go
import plotly.express as px

# 1. Örnek;
# Pasta grafiğini özelleştirme;
colors = ['blue', 'green', 'black', 'purple', 'red', 'brown']
fig = go.Figure(data=[go.Pie(labels=['Water', 'Grass', 'Normal', 'Psychic', 'Fire', 'Ground'],
                             values=[110, 90, 80, 80, 70, 60])])

# Her pasta dilimi için gezinme bilgisini, metin boyutunu, çekme miktarını ve konturu tanımlama;
fig.update_traces(
    hoverinfo='label+percent',
    textfont_size=20,
    textinfo='label+percent',
    pull=[0.1, 0, 0.2, 0, 0, 0],
    marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2))
)

# Veriyi gösterir;
fig.show()

# 2. Örnek;
# Veri setini yükle ve filtrele
df_samer = px.data.gapminder().query("year == 2007").query("continent == 'Asia'")

# Pie grafiği oluştur
fig = px.pie(
    data_frame=df_samer,
    values='pop',
    names='country',
    title='Population of Asian continent',
    color_discrete_sequence=px.colors.sequential.RdBu
)

# Grafiği göster
fig.show()
