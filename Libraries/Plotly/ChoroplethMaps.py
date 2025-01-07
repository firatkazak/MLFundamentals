import pandas as pd
import plotly.express as px
from urllib.request import urlopen  # Verilen bir URL'den veri almamızı sağlar
import json  # JSON verilerini çözümlemek için kullanılır

# ABD ilçe geometri verilerini alma;
with urlopen(url='https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response: counties = json.load(response)

# Her ilçenin Federal Bilgi İşleme numarasına göre işsizlik verilerini alma;
df = pd.read_csv(filepath_or_buffer="https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv", dtype={"fips": str})

# İlçe JSON verilerini kullanarak haritayı çizip, 12 aralığındaki işsizlik değerlerini kullanarak renklendirme;
fig = px.choropleth(data_frame=df,
                    geojson=counties,
                    locations='fips',
                    color='unemp',
                    color_continuous_scale="Viridis",
                    range_color=(0, 12),
                    scope="usa",
                    labels={'unemp': 'unemployment rate'}
                    )

# Veriyi gösterme;
fig.show()
