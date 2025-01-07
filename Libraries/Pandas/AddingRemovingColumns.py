import pandas as pd
import numpy as np

coffee = pd.read_csv(filepath_or_buffer="C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Pandas/Data/Inputs/coffee.csv")
bios = pd.read_csv(filepath_or_buffer="C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Pandas/Data/Inputs/bios.csv")

coffee["price"] = 4.99  # Ekleme => price adında bir kolon ekledik ve bütün fiyatlarını 4.99 yaptık.
coffee["new_price"] = np.where(coffee["Coffee Type"] == "Espresso", 3.99, 5.99)  # new_price adında bir kolon ekledik Espresso'yu 3.99 yaptık.
coffee.drop(columns=["price"], inplace=True)  # price isimli tabloyu kaldırdık.
coffee["revenue"] = coffee["Units Sold"] * coffee["new_price"]  # Satılan ürün sayısı ile ücretleri çarptık ve hasılatı topladık bunu revenue kolonuna ekledik.
coffee.rename(columns={"new_price": "price"})  # new_price'ı price olarak değiştirdik.

print(coffee)
#           Day Coffee Type  Units Sold  new_price  revenue
# 0      Monday    Espresso          25       3.99    99.75
# 1      Monday       Latte          15       5.99    89.85
# 2     Tuesday    Espresso          30       3.99   119.70
# 3     Tuesday       Latte          20       5.99   119.80
# 4   Wednesday    Espresso          35       3.99   139.65
# 5   Wednesday       Latte          25       5.99   149.75
# 6    Thursday    Espresso          40       3.99   159.60
# 7    Thursday       Latte          30       5.99   179.70
# 8      Friday    Espresso          45       3.99   179.55
# 9      Friday       Latte          35       5.99   209.65
# 10   Saturday    Espresso          45       3.99   179.55
# 11   Saturday       Latte          35       5.99   209.65
# 12     Sunday    Espresso          45       3.99   179.55
# 13     Sunday       Latte          35       5.99   209.65

bios_new = bios.copy()
bios_new['first_name'] = bios_new['name'].str.split(' ').str[0]  # first_name adında bir sütun ekledik. name'deki isimleri buraya aktardık.
print(bios_new.query('first_name == "Shaquille"'))  # Keith isimli olanları getirdik.
#         athlete_id              name  ... died_date first_name
# 6722          6755  Shaquille O'Neal  ...       NaN  Shaquille
# 143978      147636   Shaquille Moosa  ...       NaN  Shaquille

result = bios["height_category"] = bios["height_cm"].apply(lambda x: "Short" if x < 165 else ("Average" if x < 185 else "Tall"))

print(result)


# 0            Tall
# 1         Average
# 2         Average
# 3         Average
# 4            Tall
#            ...
# 145495    Average
# 145496    Average
# 145497      Short
# 145498    Average
# 145499       Tall
# Name: height_cm, Length: 145500, dtype: object

# METOT OLARAK YAPMA;

def categorize_athlete(row):
    if row["height_cm"] < 175 and row["weight_kg"] < 70:
        return "LightWeight"
    elif row["height_cm"] < 185 or row["weight_kg"] <= 80:
        return "MiddleWeight"
    else:
        return "HeavyWeight"


row = {"height_cm": 178, "weight_kg": 75}  # Bir sözlük oluşturduk.
category = categorize_athlete(row)
print(category)  # Çıktı: MiddleWeight

result2 = bios["Category"] = bios.apply(categorize_athlete, axis=1)
print(result2)
# 0          HeavyWeight
# 1         MiddleWeight
# 2         MiddleWeight
# 3          LightWeight
# 4          HeavyWeight
#               ...
# 145495     LightWeight
# 145496     LightWeight
# 145497     LightWeight
# 145498    MiddleWeight
# 145499     HeavyWeight
# Length: 145500, dtype: object
