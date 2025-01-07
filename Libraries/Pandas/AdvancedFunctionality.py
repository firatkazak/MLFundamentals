import pandas as pd
import numpy as np

coffee = pd.read_csv(filepath_or_buffer="C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Pandas/Data/Inputs/coffee.csv")
bios = pd.read_csv(filepath_or_buffer="C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Pandas/Data/Inputs/bios.csv")

coffee['price'] = np.where(coffee["Coffee Type"] == "Espresso", 3.99, 5.99)
coffee["revenue"] = coffee["Units Sold"] * coffee["price"]
coffee['cumsum'] = coffee['Units Sold'].cumsum()  # Cumulative sum
coffee['yesterday_revenue'] = coffee['revenue'].shift(1)  # Shift
# print(coffee)
#           Day Coffee Type  Units Sold  ...  revenue  cumsum  yesterday_revenue
# 0      Monday    Espresso          25  ...    99.75      25                NaN
# 1      Monday       Latte          15  ...    89.85      40              99.75
# 2     Tuesday    Espresso          30  ...   119.70      70              89.85
# 3     Tuesday       Latte          20  ...   119.80      90             119.70
# 4   Wednesday    Espresso          35  ...   139.65     125             119.80
# 5   Wednesday       Latte          25  ...   149.75     150             139.65
# 6    Thursday    Espresso          40  ...   159.60     190             149.75
# 7    Thursday       Latte          30  ...   179.70     220             159.60
# 8      Friday    Espresso          45  ...   179.55     265             179.70
# 9      Friday       Latte          35  ...   209.65     300             179.55
# 10   Saturday    Espresso          45  ...   179.55     345             209.65
# 11   Saturday       Latte          35  ...   209.65     380             179.55
# 12     Sunday    Espresso          45  ...   179.55     425             209.65
# 13     Sunday       Latte          35  ...   209.65     460             179.55

latte = coffee[coffee['Coffee Type'] == "Latte"].copy()  # Rolling window
latte['3day'] = latte['Units Sold'].rolling(3).sum()  # Rolling window devam
# print(latte['3day'])
# 1       NaN
# 3       NaN
# 5      60.0
# 7      75.0
# 9      90.0
# 11    100.0
# 13    105.0
# Name: 3day, dtype: float64

bios['height_rank'] = bios['height_cm'].rank(ascending=False)  # Rank
# print(bios['height_rank'])
# 0             NaN
# 1         27597.5
# 2         27597.5
# 3         83975.0
# 4             NaN
#            ...
# 145495    87188.5
# 145496    83975.0
# 145497    96305.5
# 145498    89274.5
# 145499        NaN
# Name: height_rank, Length: 145500, dtype: float64