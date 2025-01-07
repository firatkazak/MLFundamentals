import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Model ve Tokenizer'ı yükle
model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
tokenizer = BertTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
model = BertForSequenceClassification.from_pretrained(model_name)

# Örnek metin
text = "I love programming with Python!"

# Metni token'lara ayır
inputs = tokenizer(text, return_tensors='pt')

# Model ile tahmin yap
with torch.no_grad():
    outputs = model(**inputs)

# Sonuçları al
logits = outputs.logits
predicted_class = torch.argmax(logits, dim=1)

# Sonucu yazdır
print(f'Tahmin edilen sınıf: {predicted_class.item()}')

# Sınıfın anlamını yazdır
class_labels = {
    0: "Negatif",
    1: "Nötr",
    2: "Pozitif",
    3: "Çok Pozitif",
    4: "Çok Çok Pozitif"
}
print(f'Tahmin edilen duygu: {class_labels[predicted_class.item()]}')
