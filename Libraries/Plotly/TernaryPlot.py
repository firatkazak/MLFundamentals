import plotly.express as px

# Veriyi çekme;
df_exp = px.data.experiment()
# Veriyi işleme;
fig = px.scatter_ternary(data_frame=df_exp, a="experiment_1", b="experiment_2", c='experiment_3', hover_name="group", color="gender")
# Veriyi gösterme;
fig.show()
