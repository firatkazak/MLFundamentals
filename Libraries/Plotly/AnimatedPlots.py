import plotly.express as px

df_cnt = px.data.gapminder()

fig1 = px.scatter(data_frame=df_cnt,
                  x="gdpPercap",
                  y="lifeExp",
                  animation_frame="year",
                  animation_group="country",
                  size="pop",
                  color="continent",
                  hover_name="country",
                  log_x=True,
                  size_max=55,
                  range_x=[100, 100000],
                  range_y=[25, 90]
                  )

fig2 = px.bar(data_frame=df_cnt,
              x="continent",
              y="pop",
              color="continent",
              animation_frame="year",
              animation_group="country",
              range_y=[0, 4000000000]
              )

# Veriyi gösterme;
fig1.show()
fig2.show()
