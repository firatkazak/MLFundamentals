import pandas as pd

bios = pd.read_csv(filepath_or_buffer='C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Pandas/Data/Inputs/bios.csv')

print(bios.loc[bios["height_cm"] > 225, ["name", "height_cm"]])  # 225 cm'den uzun olanları listeledik. Sadece boy ve ismi ver dedik. Yao Ming çıktı.
#            name  height_cm
# 89070  Yao Ming      226.0

print(bios[bios["height_cm"] > 225][["name", "height_cm"]])  # Yukarıdakinin aynısı, yöntem farklı.
#            name  height_cm
# 89070  Yao Ming      226.0

print(bios[(bios["height_cm"] > 215) & (bios["born_country"] == "USA")][["name", "height_cm"]])  # ABD'li 215'ten uzun kişileri getir dedik.
#                     name  height_cm
# 5781      Tommy Burleson      223.0
# 6722    Shaquille O'Neal      216.0
# 6937      David Robinson      216.0
# 123850    Tyson Chandler      216.0

print(bios[bios["name"].str.contains("Shaquille")])  # Shaquille ismindeki oyuncuları getirdik.
#         athlete_id              name  ... weight_kg died_date
# 6722          6755  Shaquille O'Neal  ...     137.0       NaN
# 143978      147636   Shaquille Moosa  ...       NaN       NaN

print(bios[bios['name'].str.contains('shaquille|iverson', case=False)])  # shaquille ya da iverson olanları getirecek.
#         athlete_id              name  ... weight_kg died_date
# 6722          6755  Shaquille O'Neal  ...     137.0       NaN
# 107420      108545     Allen Iverson  ...      83.0       NaN
# 111446      112686    Dennis Iverson  ...      73.0       NaN
# 143978      147636   Shaquille Moosa  ...       NaN       NaN

print(bios[bios["born_country"].isin(["USA", "FRA", "GBR"])])  # Fransa, ABD ya da Britanya doğumlu olanları getirecek.
#         athlete_id               name  ... weight_kg   died_date
# 4                5       Albert Canet  ...       NaN  1930-07-25
# 37              38    Helen Aitchison  ...       NaN  1947-05-26
# 38              39  Geraldine Beamish  ...       NaN  1972-05-10
# 39              40       Dora Boothby  ...       NaN  1970-02-22
# 40              41     Julie Bradbury  ...      64.0         NaN
# Çok uzun bir kısmını yazdırdım.

print(bios[bios["born_country"].isin(["USA", "FRA", "GBR"]) & (bios["name"].str.startswith("Shaquille"))])  # Shaquille isimli USA, FR yada GBR doğumlular.
#       athlete_id              name   born_date  ... height_cm weight_kg died_date
# 6722        6755  Shaquille O'Neal  1972-03-06  ...     216.0     137.0       NaN

print(bios.query(' born_country == "USA" and born_city == "Seattle" '))  # Doğduğu ülke USA, şehir Seattle olanları sorguladık.
#         athlete_id                   name  ... weight_kg   died_date
# 11030        11088          David Halpern  ...      79.0         NaN
# 12800        12870            Todd Trewin  ...      75.0         NaN
# 15476        15583         Scott McKinley  ...      75.0         NaN
# 29079        29293            Joyce Tanac  ...      49.0         NaN
# 31135        31371        Bill Kuhlemeier  ...       NaN  2001-07-08
# ...            ...                    ...  ...       ...         ...
# 133392      136331          Hans Struzyna  ...      91.0         NaN
# 135448      138662  Maude Davis Crossland  ...       NaN         NaN
# 136993      140229        Jenell Berhorst  ...       NaN         NaN
# 143507      147159         Nevin Harrison  ...      73.0         NaN
# 145446      149169       Corinne Stoddard  ...       NaN         NaN
