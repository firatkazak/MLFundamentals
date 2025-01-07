# Import the requisite library
import spacy

# Build upon the spaCy Small Model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "The village of Treblinka is in Poland. Treblinka was also an extermination camp."

# Create the Doc object
doc = nlp(text)

# extract entities
for ent in doc.ents:
    print(ent.text, ent.label_)
# Treblinka GPE
# Poland GPE

# Import the requisite library
import spacy

# Build upon the spaCy Small Model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "The village of Treblinka is in Poland. Treblinka was also an extermination camp."

# Create the EntityRuler
ruler = nlp.add_pipe("entity_ruler")

# List of Entities and Patterns
patterns = [
    {"label": "GPE", "pattern": "Treblinka"}
]

ruler.add_patterns(patterns)

doc = nlp(text)

# extract entities
for ent in doc.ents:
    print(ent.text, ent.label_)
# Treblinka GPE
# Poland GPE
# Treblinka GPE

print(nlp.analyze_pipes())
# {'summary': {'tok2vec': {'assigns': ['doc.tensor'], 'requires': [], 'scores': [], 'retokenizes': False}, 'tagger': {'assigns': ['token.tag'], 'requires': [], 'scores': ['tag_acc'], 'retokenizes': False}, 'parser': {'assigns': ['token.dep', 'token.head', 'token.is_sent_start', 'doc.sents'], 'requires': [], 'scores': ['dep_uas', 'dep_las', 'dep_las_per_type', 'sents_p', 'sents_r', 'sents_f'], 'retokenizes': False}, 'attribute_ruler': {'assigns': [], 'requires': [], 'scores': [], 'retokenizes': False}, 'lemmatizer': {'assigns': ['token.lemma'], 'requires': [], 'scores': ['lemma_acc'], 'retokenizes': False}, 'ner': {'assigns': ['doc.ents', 'token.ent_iob', 'token.ent_type'], 'requires': [], 'scores': ['ents_f', 'ents_p', 'ents_r', 'ents_per_type'], 'retokenizes': False}, 'entity_ruler': {'assigns': ['doc.ents', 'token.ent_type', 'token.ent_iob'], 'requires': [], 'scores': ['ents_f', 'ents_p', 'ents_r', 'ents_per_type'], 'retokenizes': False}}, 'problems': {'tok2vec': [], 'tagger': [], 'parser': [], 'attribute_ruler': [], 'lemmatizer': [], 'ner': [], 'entity_ruler': []}, 'attrs': {'token.head': {'assigns': ['parser'], 'requires': []}, 'token.ent_type': {'assigns': ['ner', 'entity_ruler'], 'requires': []}, 'doc.sents': {'assigns': ['parser'], 'requires': []}, 'token.tag': {'assigns': ['tagger'], 'requires': []}, 'token.dep': {'assigns': ['parser'], 'requires': []}, 'doc.tensor': {'assigns': ['tok2vec'], 'requires': []}, 'doc.ents': {'assigns': ['ner', 'entity_ruler'], 'requires': []}, 'token.ent_iob': {'assigns': ['ner', 'entity_ruler'], 'requires': []}, 'token.is_sent_start': {'assigns': ['parser'], 'requires': []}, 'token.lemma': {'assigns': ['lemmatizer'], 'requires': []}}}

# Build upon the spaCy Small Model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "The village of Treblinka is in Poland. Treblinka was also an extermination camp."

# Create the EntityRuler
ruler = nlp.add_pipe(factory_name="entity_ruler", after="ner")

# List of Entities and Patterns
patterns = [
    {"label": "GPE", "pattern": "Treblinka"}
]

ruler.add_patterns(patterns)

doc = nlp(text)

# extract entities
for ent in doc.ents:
    print(ent.text, ent.label_)
# Treblinka GPE
# Poland GPE
# Treblinka GPE

# Import the requisite library
import spacy

# Sample text
text = "This is a sample number (555) 555-5555."

# Build upon the spaCy Small Model
nlp = spacy.blank("en")

# Create the Ruler and Add it
ruler = nlp.add_pipe("entity_ruler")

# List of Entities and Patterns (source: https://spacy.io/usage/rule-based-matching)
patterns = [
    {"label": "PHONE_NUMBER", "pattern": [{"ORTH": "("}, {"SHAPE": "ddd"}, {"ORTH": ")"}, {"SHAPE": "ddd"},
                                          {"ORTH": "-", "OP": "?"}, {"SHAPE": "dddd"}]}
]
# add patterns to ruler
ruler.add_patterns(patterns)

# create the doc
doc = nlp(text)

# extract entities
for ent in doc.ents:
    print(ent.text, ent.label_)
# (555) 555-5555 PHONE_NUMBER
