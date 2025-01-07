import spacy
import requests
from bs4 import BeautifulSoup

# spaCy pipeline
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

# Veri çekme ve BeautifulSoup kullanımı
s = requests.get("https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt")
# Parser'ı açıkça belirtelim
soup = BeautifulSoup(s.content, 'html.parser').text.replace("-\n", "").replace("\n", " ")

# spaCy maksimum uzunluk ayarı
nlp.max_length = 5278439

# Metin işleme
doc = nlp(soup)
print(len(list(doc.sents)))

# İkinci spaCy modelini yükleme
nlp2 = spacy.load("en_core_web_sm")
nlp2.max_length = 5278439

# Metin işleme
doc = nlp2(soup)
print(len(list(doc.sents)))

# Pipeline analizini kontrol etme
nlp2.analyze_pipes()
