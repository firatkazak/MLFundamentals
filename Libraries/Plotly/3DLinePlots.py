import plotly.express as px
import seaborn as sns

# Veriyi çekme;
flights = sns.load_dataset("flights")

# Uçuş verilerini kullanarak 3 boyutlu bir dağılım grafiği oluşturma;
fig = px.line_3d(data_frame=flights, x='year', y='month', z='passengers', color='year')

# Veriyi gösterme;
fig.show()
