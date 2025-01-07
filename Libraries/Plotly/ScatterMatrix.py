import plotly.express as px
import seaborn as sns

# Veriyi çekme;
flights = sns.load_dataset("flights")

# Uçuş verilerini kullanarak 3 boyutlu bir dağılım grafiği oluşturma;
fig = px.scatter_matrix(flights, color='month')

# Veriyi gösterme;
fig.show()
