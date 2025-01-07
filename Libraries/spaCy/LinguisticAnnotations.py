import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")

with open("C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/spaCy/Data/Inputs/wiki_us.txt", "r") as f:
    text = f.read()

print(text)

doc = nlp(text)

print(doc)

print(len(doc))
print(len(text))

for token in text[:10]:
    print(token)

for token in doc[:10]:
    print(token)

for token in text.split()[:10]:
    print(token)

words = text.split()[:10]

i = 5
for token in doc[i:8]:
    print(f"SpaCy Token {i}:\n{token}\nWord Split {i}:\n{words[i]}\n\n")
    i = i + 1

for sent in doc.sents:
    print(sent)

sentence1 = list(doc.sents)[0]
print(sentence1)

token2 = sentence1[2]
print(token2)

print(token2.text)

print(token2.head)

print(token2.left_edge)

print(token2.right_edge)

print(token2.ent_type)

print(token2.ent_type_)

print(token2.ent_iob_)

print(token2.lemma_)

print(sentence1[12].lemma_)

print(sentence1[12].morph)

print(token2.pos_)

print(token2.dep_)

print(token2.lang_)

for token in sentence1:
    print(token.text, token.pos_, token.dep_)

displacy.render(sentence1, style="dep")

for ent in doc.ents:
    print(ent.text, ent.label_)

displacy.render(doc, style="ent")
