import numpy as np
import pandas as pd

coffee = pd.read_csv(filepath_or_buffer='C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Pandas/Data/Inputs/coffee.csv')
coffee.loc[[0, 1], "Units Sold"] = np.nan  # 1. ve 2. elemanın Units Sold değerini NaN yaptık. Yani sıfırladık.
coffee = coffee.fillna(coffee["Units Sold"].mean())  # NaN olan değerleri doldurduk. mean metodu ortalamayı alır, o yüzden ortalama fiyat ekledi.
coffee = coffee.fillna(coffee["Units Sold"].interpolate())  # NaN değerlerini bir enterpolasyon yöntemi kullanarak doldurur.
coffee.dropna(subset=["Units Sold"], inplace=True)  # Units Sold'u NaN olan elemanları kaldırıyor.

print(coffee[coffee["Units Sold"].isna()])  # nan olanları getirir.
# Empty DataFrame
# Columns: [Day, Coffee Type, Units Sold]
# Index: []

print(coffee[coffee["Units Sold"].notna()])  # nan olmayanları getirir.
#           Day Coffee Type  Units Sold
# 0      Monday    Espresso        35.0
# 1      Monday       Latte        35.0
# 2     Tuesday    Espresso        30.0
# 3     Tuesday       Latte        20.0
# 4   Wednesday    Espresso        35.0
# 5   Wednesday       Latte        25.0
# 6    Thursday    Espresso        40.0
# 7    Thursday       Latte        30.0
# 8      Friday    Espresso        45.0
# 9      Friday       Latte        35.0
# 10   Saturday    Espresso        45.0
# 11   Saturday       Latte        35.0
# 12     Sunday    Espresso        45.0
# 13     Sunday       Latte        35.0
