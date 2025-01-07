import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Iris veri setini kullanıyoruz;
df_iris = px.data.iris()

# Sağlanan sütun sayısı için x, y, farklı renk, sağlanan sütuna göre boyut ve üzerine gelindiğinde görüntülenecek ek veriler tanımlayarak bir dağılım grafiği oluşturduk;
px.scatter(data_frame=df_iris,
           x="sepal_width",
           y="sepal_length",
           color="species",
           size='petal_length',
           hover_data=['petal_width']
           )

# Çizgi genişliği 2 olan, opak ve genişliğe göre renklendirilmiş siyah işaretleyici kenarlarıyla özelleştirilmiş bir dağılım oluşturun. Ayrıca sağ tarafta bir ölçek gösterin.
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_iris.sepal_width,
    y=df_iris.sepal_length,
    mode='markers',
    marker_color=df_iris.sepal_width,
    text=df_iris.species,
    marker=dict(showscale=True)
))

fig.update_traces(marker_line_width=2, marker_size=10)

# Çok sayıda veri ile çalışırken Scattergl kullanıyoruz;
fig = go.Figure(data=go.Scattergl(
    x=np.random.randn(100000),
    y=np.random.randn(100000),
    mode='markers',
    marker=dict(
        color=np.random.randn(100000),
        colorscale='Viridis',
        line_width=1
    )
))

# Veriyi gösterir;
fig.show()

# NOT:
