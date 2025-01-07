import plotly.express as px
import seaborn as sns

# Veriyi çekme;
flights = sns.load_dataset("flights")

# Uçuş verilerini kullanarak 3 boyutlu bir dağılım grafiği oluşturma;
fig = px.scatter_3d(data_frame=flights,
                    x='year',
                    y='month',
                    z='passengers',
                    color='year',
                    opacity=0.7,
                    width=800,
                    height=400
                    )

# Veriyi gösterme;
fig.show()
