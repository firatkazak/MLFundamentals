import plotly.express as px

# ABD verileri için sorgulama yaparak ABD'deki nüfus değişimini alma;
df_us = px.data.gapminder().query("country == 'United States'")
px.bar(df_us, x='year', y='pop')

# Daha fazla özelleştirme ile sıralı bir çubuk oluşturma;
df_tips = px.data.tips()
px.bar(df_tips, x='day', y='tip', color='sex', title='Tips by Sex on Each Day',
       labels={'tip': 'Tip Amount', 'day': 'Day of the Week'})

# Çubukları yan yana yerleştirme;
px.bar(df_tips, x="sex", y="total_bill",
       color='smoker', barmode='group')

# Avrupa'da 2007 yılında 2000000'den büyük ülkeler için nüfus verilerini görüntüle;
df_europe = px.data.gapminder().query("continent == 'Europe' and year == 2007 and pop > 2.e6")
fig = px.bar(df_europe, y='pop', x='country', text='pop', color='country')

# Bar toplam değerini 2 hassasiyet değerine sahip barların üzerine koyma;
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

# Fontsize ve uniformtext_mode='hide' değerlerini ayarlayarak, metin sığmazsa gizlenmesini söyler;
fig.update_layout(uniformtext_minsize=8)

# Etiketleri 45 derece döndürme;
fig.update_layout(xaxis_tickangle=-45)

# Veriyi gösterir;
fig.show()

# NOT: Çubukların üstüne gelince ülke adı ve toplam nüfus yazıyor.
