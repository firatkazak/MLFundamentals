import matplotlib.pyplot as plt

langs = ["Python", "C++", "Java", "C#", "Go"]
votes = [50, 24, 14, 6, 17]
explodes = [0, 0, 0, 0.2, 0]
plt.pie(
    x=votes, # verilen oy oranı.
    labels=langs, # programlama dilleri.
    explode=explodes,  # parçalara ayırdığımız yer.
    autopct="%.2f%%",  # parçalara ayırdığımız pastayı dışarı çıkarttığımız yer. burada c# çıkmış.
    pctdistance=1.3,  # c# 5.41% yazan yerlerin arasındaki boşluk.
    startangle=90  # başlangıç noktası. Burada saat 12 noktası, 90derece verilmiş yani.
)

picture_path = "C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Matplotlib/Data/Outputs"
file_name = "pie_chart.png"
plt.savefig(f"{picture_path}/{file_name}", dpi=300, bbox_inches='tight')
plt.close()
