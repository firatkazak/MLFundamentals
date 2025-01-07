import plotly.express as px
import plotly.graph_objects as go

# Tek bir grafik oluşturmak için Google fiyat verilerini kullanın;
df_stocks = px.data.stocks()
px.line(df_stocks, x='date', y='GOOG', labels={'x': 'Date', 'y': 'Price'})

# Çoklu çizgi grafikleri yapma;
px.line(df_stocks, x='date', y=['GOOG', 'AAPL'], labels={'x': 'Date', 'y': 'Price'}, title='Apple Vs. Google')

# Üzerine grafikler ekleyeceğimiz bir şekil yaratma;
fig = go.Figure()

# Veri setinden tek tek veri sütunlarını çekebilir ve işaretleyici kullanabiliriz;
fig.add_trace(go.Scatter(x=df_stocks.date, y=df_stocks.AAPL, mode='lines', name='Apple'))
fig.add_trace(go.Scatter(x=df_stocks.date, y=df_stocks.AMZN, mode='lines+markers', name='Amazon'))

# Özel çizgiler oluşturabiliriz(Dashes : dash, dot, dashdot);
fig.add_trace(go.Scatter(x=df_stocks.date, y=df_stocks.GOOG, mode='lines+markers', name='Google',
                         line=dict(color='firebrick', width=2, dash='dashdot')))

# Figürün daha fazla stillendirilmesi;
fig.update_layout(title='Stock Price Data 2018 - 2020',
                  xaxis_title='Price', yaxis_title='Date')

# Gri çizgileri, yazı tiplerini, çizgi genişliklerini ve daha fazlasını ızgara olmadan gösterir;
fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        ),
    ),

    # Y eksenindeki her şeyi kapatma;
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showline=False,
        showticklabels=False,
    ),
    autosize=False,
    margin=dict(
        autoexpand=False,
        l=100,
        r=20,
        t=110,
    ),
    showlegend=False,
    plot_bgcolor='white'
)

# Çıktıyı alma;
fig.show()

# : Turuncu Amazon, Mavi Apple, Bordo Google'ı temsil ediyor. Resim aslında hareketli, üstüne geldiğinde tarihi ve hangi şirket olduğunu gösteriyor.
