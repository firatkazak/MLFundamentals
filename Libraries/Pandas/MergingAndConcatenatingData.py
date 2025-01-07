import pandas as pd

coffee = pd.read_csv(filepath_or_buffer='C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Pandas/Data/Inputs/coffee.csv')  # Load data from CSV
nocs = pd.read_csv(filepath_or_buffer='C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Pandas/Data/Inputs/noc_regions.csv')  # Load data from CSV
bios = pd.read_csv(filepath_or_buffer='C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Pandas/Data/Inputs/bios.csv')  # Load data from CSV

bios_new = pd.merge(bios, nocs, left_on="born_country", right_on="NOC", how="left", suffixes=['bios', 'nocdf'])
bios_new.rename(columns={"region": "born_country_full"}, inplace=True)

# Concatenate DataFrames
usa = bios[bios['born_country'] == 'USA'].copy()
gbr = bios[bios['born_country'] == 'GBR'].copy()
new_df = pd.concat([usa, gbr])
print(new_df)
#         athlete_id                name  ... weight_kg   died_date
# 54              55       Monique Javer  ...      64.0         NaN
# 960            964    Xóchitl Escobedo  ...      60.0         NaN
# 961            965   Angélica Gavaldón  ...      54.0         NaN
# 1231          1238      Bert Schneider  ...       NaN  1986-02-20
# 1345          1352          Laura Berg  ...      61.0         NaN
# ...            ...                 ...  ...       ...         ...
# 144811      148512  Benjamin Alexander  ...       NaN         NaN
# 144815      148517       Ashley Watson  ...       NaN         NaN
# 145005      148716     Peder Kongshaug  ...      86.0         NaN
# 145319      149041          Axel Brown  ...       NaN         NaN
# 145388      149111      Jean-Luc Baker  ...       NaN         NaN
#
# [15433 rows x 10 columns]
