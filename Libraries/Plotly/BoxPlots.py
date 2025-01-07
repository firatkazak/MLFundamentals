import plotly.graph_objects as go
import plotly.express as px

# Veriyi çekme;
df_tips = px.data.tips()

# Adding standard deviation and mean
fig = go.Figure()

# Complex Styling
df_stocks = px.data.stocks()
fig = go.Figure()

# Show all points, spread them so they don't overlap and change whisker width
fig.add_trace(go.Box(y=df_stocks.GOOG, boxpoints='all', name='Google', fillcolor='blue', jitter=0.5, whiskerwidth=0.2))
fig.add_trace(go.Box(y=df_stocks.AAPL, boxpoints='all', name='Apple', fillcolor='red', jitter=0.5, whiskerwidth=0.2))

# Change background / grid colors
fig.update_layout(title='Google vs. Apple',
                  yaxis=dict(gridcolor='rgb(255, 255, 255)',
                             gridwidth=3),
                  paper_bgcolor='rgb(243, 243, 243)',
                  plot_bgcolor='rgb(243, 243, 243)')

# Grafiği göster
fig.show()

# NOT: Bir kutu grafiği farklı değişkenleri karşılaştırmanıza olanak tanır. Kutu, verilerin dörtte birlik kısımlarını gösterir. Ortadaki çubuk medyandır. Bıyıklar, aykırı değer olarak kabul edilen noktalar haricindeki tüm diğer verilere uzanır.

df_tips = px.data.tips()
px.box(df_tips, x='sex', y='tip', points='all')
px.box(df_tips, x='day', y='tip', color='sex')
fig = go.Figure()
fig.add_trace(go.Box(x=df_tips.sex, y=df_tips.tip, marker_color='blue', boxmean='sd'))
fig.show()
