import spacy
import numpy as np

nlp = spacy.load("en_core_web_md")

with open("C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/spaCy/Data/Inputs/wiki_us.txt", "r") as f:
    text = f.read()

doc = nlp(text)
sentence1 = list(doc.sents)[0]

print(sentence1[0].vector)
# https://stackoverflow.com/questions/54717449/mapping-word-vector-to-the-most-similar-closest-word-using-spacy
your_word = "dog"

ms = nlp.vocab.vectors.most_similar(np.asarray([nlp.vocab.vectors[nlp.vocab.strings[your_word]]]), n=10)
words = [nlp.vocab.strings[w] for w in ms[0][0]]
distances = ms[2]
print(words)

nlp = spacy.load("en_core_web_md")  # make sure to use larger package!
doc1 = nlp("I like salty fries and hamburgers.")
doc2 = nlp("Fast food tastes very good.")

# Similarity of two documents
print(doc1, "<->", doc2, doc1.similarity(doc2))

# Similarity of tokens and spans
french_fries = doc1[2:4]
burgers = doc1[5]
print(french_fries, "<->", burgers, french_fries.similarity(burgers))
