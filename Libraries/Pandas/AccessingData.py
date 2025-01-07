import pandas as pd

coffee = pd.read_csv(filepath_or_buffer='C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Pandas/Data/Inputs/coffee.csv')  # Load data from CSV

print(coffee.head())  # İlk 5'i verir. İçine sayı yazarsan yazdığın sayı kadarını verir.
#          Day Coffee Type  Units Sold
# 0     Monday    Espresso          25
# 1     Monday       Latte          15
# 2    Tuesday    Espresso          30
# 3    Tuesday       Latte          20
# 4  Wednesday    Espresso          35

print(coffee.tail())  # Son 5'i verir. İçine sayı yazarsan yazdığın sayı kadarını verir.
#          Day Coffee Type  Units Sold
# 9     Friday       Latte          35
# 10  Saturday    Espresso          45
# 11  Saturday       Latte          35
# 12    Sunday    Espresso          45
# 13    Sunday       Latte          35

print(coffee.sample())  # Örnek veri verir. İçine sayı yazarsan yazdığın sayı kadarını verir.
#        Day Coffee Type  Units Sold
# 13  Sunday       Latte          35

print(coffee.loc[0])  # Verilen rakamdaki Index'e erişir. Burada örnek 0.
# Day              Monday
# Coffee Type    Espresso
# Units Sold           25
# Name: 0, dtype: object

print(coffee.iloc[0])  # Verilen rakamdaki Index'e erişir. Burada örnek 0.
# Day              Monday
# Coffee Type    Espresso
# Units Sold           25
# Name: 0, dtype: object

print(coffee.iat[0, 0])  # Bir satır/sütun çifti için tek bir değere tamsayı konumuna göre erişir.
# Monday

print(coffee.at[0, "Units Sold"])  # Bir satır/sütun etiket çifti için tek bir değere erişir.
# 25

print(coffee["Units Sold"])  # Yazılan sütundaki değerleri verir. Burada örnek Units Sold sütunu.
# 0     25
# 1     15
# 2     30
# 3     20
# 4     35
# 5     25
# 6     40
# 7     30
# 8     45
# 9     35
# 10    45
# 11    35
# 12    45
# 13    35
# Name: Units Sold, dtype: int64

print(coffee.sort_values(by="Units Sold", ascending=False))  # Verilen değeri azdan çoka sıralar. Çoktan aza sıralamak için ascending False yapılır.
#           Day Coffee Type  Units Sold
# 8      Friday    Espresso          45
# 10   Saturday    Espresso          45
# 12     Sunday    Espresso          45
# 6    Thursday    Espresso          40
# 4   Wednesday    Espresso          35
# 9      Friday       Latte          35
# 11   Saturday       Latte          35
# 13     Sunday       Latte          35
# 2     Tuesday    Espresso          30
# 7    Thursday       Latte          30
# 0      Monday    Espresso          25
# 5   Wednesday       Latte          25
# 3     Tuesday       Latte          20
# 1      Monday       Latte          15

# iterrows metodu ile veride iterasyon dönme;
for index, row in coffee.iterrows():
    print(index)
    print(row["Units Sold"])  # Eğer spesifik bir şey istiyorsak [] ile belirtiyoruz. Burada örnek Units Sold.
    print("\n")
# 0
# 25
#
# 1
# 15
#
# 2
# 30
#
# 3
# 20
#
# 4
# 35
#
# 5
# 25
#
# 6
# 40
#
# 7
# 30
#
# 8
# 45
#
# 9
# 35
#
# 10
# 45
#
# 11
# 35
#
# 12
# 45
#
# 13
# 35
