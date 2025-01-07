import plotly.express as px

df_wind = px.data.wind()

# 1. Durum;
fig1 = px.scatter_polar(df_wind, r="frequency", theta="direction", color="strength", size="frequency", symbol="strength")

# Veriler ayrıca radyal çizgiler kullanılarak da çizilebilir. Bir şablon, verilerin daha kolay görülmesini sağlar;
fig2 = px.line_polar(df_wind, r="frequency", theta="direction", color="strength", line_close=True, template="plotly_dark", width=800, height=400)

# Veriyi gösterme;
fig1.show()
fig2.show()
