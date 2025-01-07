import pandas as pd

# Örnek bir veri çerçevesi oluşturalım
data = {'Ad': ['Ali', 'Veli', 'Ayşe', 'Fatma'],
        'Yaş': [25, 30, 22, 28],
        'Şehir': ['Ankara', 'İstanbul', 'İzmir', 'Ankara']}
df = pd.DataFrame(data)

# Veri çerçevesini ekrana yazdıralım
print(df)

# Yaşı 25'ten büyük olanları filtreleyelim
filtered_df = df[df['Yaş'] > 25]
print(filtered_df)

filtered_df.to_csv('C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Pandas/Data/Outputs/filtered_data.csv', index=False)
filtered_df.to_excel('C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Pandas/Data/Outputs/filtered_data.xlsx', index=False)
