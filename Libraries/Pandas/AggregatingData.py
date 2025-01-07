import pandas as pd
import numpy as np

coffee = pd.read_csv(filepath_or_buffer="C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Pandas/Data/Inputs/coffee.csv")
bios = pd.read_csv(filepath_or_buffer="C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Pandas/Data/Inputs/bios.csv")

coffee['Units Sold'] = coffee['Units Sold'].fillna(0)  # NaN'leri 0 ile doldur
coffee['Units Sold'] = coffee['Units Sold'].interpolate()  # Eksik değerleri interpolasyon ile doldur
coffee = coffee.dropna(subset=['Units Sold'])  # NaN içeren satırları kaldır

coffee.groupby(["Coffee Type"])["Units Sold"].sum()  #
coffee.groupby(["Coffee Type"])["Units Sold"].mean()  #

coffee['price'] = np.where(coffee["Coffee Type"] == "Espresso", 3.99, 5.99)
coffee["revenue"] = coffee["Units Sold"] * coffee["price"]
coffee.groupby(['Coffee Type', 'Day']).agg({'Units Sold': 'sum', 'price': 'mean'})  # Günleri ve Kahve tiplerini getiriyor. Ayrıca Units Sold'un toplamasını, price'ın ortalamasını hesaplıyor.
coffee.pivot(columns='Coffee Type', index='Day', values='revenue')  #
# print(coffee)
#           Day Coffee Type  Units Sold  price  revenue
# 0      Monday    Espresso          25   3.99    99.75
# 1      Monday       Latte          15   5.99    89.85
# 2     Tuesday    Espresso          30   3.99   119.70
# 3     Tuesday       Latte          20   5.99   119.80
# 4   Wednesday    Espresso          35   3.99   139.65
# 5   Wednesday       Latte          25   5.99   149.75
# 6    Thursday    Espresso          40   3.99   159.60
# 7    Thursday       Latte          30   5.99   179.70
# 8      Friday    Espresso          45   3.99   179.55
# 9      Friday       Latte          35   5.99   209.65
# 10   Saturday    Espresso          45   3.99   179.55
# 11   Saturday       Latte          35   5.99   209.65
# 12     Sunday    Espresso          45   3.99   179.55
# 13     Sunday       Latte          35   5.99   209.65

# print(bios[bios["born_country"] == "USA"]["born_region"].value_counts())  # Doğduğu ülke USA olanları getir dedik.
bios['born_date'] = pd.to_datetime(bios['born_date'], errors='coerce')  # Hatalı değerleri NaT olarak işaretler
bios.groupby(bios['born_date'].dt.year)['name'].count().reset_index().sort_values('name', ascending=False)  # doğum yılında en çok isim olanları sıraladı. Mesela 1973'te 1525 tane aynı addan var.

# ÇOKLU Data Gruplama;
bios['born_date'] = pd.to_datetime(bios['born_date'])
bios['month_born'] = bios['born_date'].dt.month
bios['year_born'] = bios['born_date'].dt.year
bios.groupby([bios['year_born'], bios['month_born']])['name'].count().reset_index().sort_values('name', ascending=False)
