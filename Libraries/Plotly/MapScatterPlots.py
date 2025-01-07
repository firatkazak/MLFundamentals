import plotly.express as px

# Veri çekme;
df = px.data.gapminder().query("year == 2007")
# Coğrafi veri çekme;
fig = px.scatter_geo(df, locations="iso_alpha", color="continent", hover_name="country", size="pop", projection="orthographic")
# Veriyi gösterme;
fig.show()
