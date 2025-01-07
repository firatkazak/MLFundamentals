import plotly.express as px
import numpy as np

# 2 zar atılmasına dayalı histogram çizimi;
dice_1 = np.random.randint(1, 7, 5000)
dice_2 = np.random.randint(1, 7, 5000)
dice_sum = dice_1 + dice_2

# bins yapılacak bar sayısını ve onun görsel ayarlarını yapma;
fig = px.histogram(dice_sum, nbins=11, labels={'value': 'Dice Roll'}, title='5000 Dice Roll Histogram',
                   marginal='violin', color_discrete_sequence=['green'])
fig.update_layout(xaxis_title_text='Dice Roll', yaxis_title_text='Dice Sum', bargap=0.2, showlegend=False)

# Farklı sütun verilerine dayalı yığın histogramı;
df_tips = px.data.tips()
px.histogram(df_tips, x="total_bill", color="sex")

# Grafiği göster
fig.show()
