import plotly.express as px
import seaborn as sns

# Çok sayıda subplots yaratabiliriz;
df_tips = px.data.tips()
px.scatter(data_frame=df_tips, x="total_bill", y="tip", color="smoker", facet_col="sex")

# Verileri satırlar ve sütunlar halinde sıralayabiliriz;
px.histogram(data_frame=df_tips,
             x="total_bill",
             y="tip",
             color="sex",
             facet_row="time",
             facet_col="day",
             category_orders={"day": ["Thur", "Fri", "Sat", "Sun"], "time": ["Lunch", "Dinner"]}
             )

# Bu veri çerçevesi, test sırasında gösterdikleri dikkat düzeyine göre farklı öğrencilere puanlar sağlar.
att_df = sns.load_dataset("attention")
fig = px.line(data_frame=att_df,
              x='solutions',
              y='score',
              facet_col='subject',
              facet_col_wrap=5,
              title='Scores Based on Attention'
              )

# Veriyi gösterme;
fig.show()
