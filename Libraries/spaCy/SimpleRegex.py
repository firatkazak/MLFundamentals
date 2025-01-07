import re

pattern = r"((\d){1,2} (January|February|March|April|May|June|July|August|September|October|November|December))"

text = "This is a date 2 February. Another date would be 14 August."
matches = re.findall(pattern, text)
print(matches)

text = "This is a date February 2. Another date would be 14 August."
matches = re.findall(pattern, text)
print(matches)

pattern = r"(((\d){1,2}( (January|February|March|April|May|June|July|August|September|October|November|December)))|(((January|February|March|April|May|June|July|August|September|October|November|December) )(\d){1,2}))"

text = "This is a date February 2. Another date would be 14 August."
matches = re.findall(pattern, text)
print(matches)

text = "This is a date February 2. Another date would be 14 August."
iter_matches = re.finditer(pattern, text)
print(iter_matches)

text = "This is a date February 2. Another date would be 14 August."
iter_matches = re.finditer(pattern, text)
print(iter_matches)
for hit in iter_matches:
    print(hit)

text = "This is a date February 2. Another date would be 14 August."
iter_matches = re.finditer(pattern, text)
for hit in iter_matches:
    start = hit.start()
    end = hit.end()
    print(text[start:end])

# Import the requisite library
import spacy

# Sample text
text = "This is a sample number 555-5555."

# Build upon the spaCy Small Model
nlp = spacy.blank("en")

# Create the Ruler and Add it
ruler = nlp.add_pipe("entity_ruler")

# List of Entities and Patterns (source: https://spacy.io/usage/rule-based-matching)
patterns = [
    {"label": "PHONE_NUMBER", "pattern": [{"SHAPE": "ddd"},
                                          {"ORTH": "-", "OP": "?"}, {"SHAPE": "dddd"}]}
]
# add patterns to ruler
ruler.add_patterns(patterns)

# create the doc
doc = nlp(text)

# extract entities
for ent in doc.ents:
    print(ent.text, ent.label_)

pattern = r"((\d){3}-(\d){4})"
text = "This is a sample number 555-5555."
matches = re.findall(pattern, text)
print(matches)

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
    {
        "label": "PHONE_NUMBER", "pattern": [{"TEXT": {"REGEX": "((\d){3}-(\d){4})"}}
                                             ]
    }
]
# add patterns to ruler
ruler.add_patterns(patterns)

# create the doc
doc = nlp(text)

# extract entities
for ent in doc.ents:
    print(ent.text, ent.label_)

#Import the requisite library
import spacy

#Sample text
text = "This is a sample number 5555555."
#Build upon the spaCy Small Model
nlp = spacy.blank("en")

#Create the Ruler and Add it
ruler = nlp.add_pipe("entity_ruler")

#List of Entities and Patterns (source: https://spacy.io/usage/rule-based-matching)
patterns = [
                {
                    "label": "PHONE_NUMBER", "pattern": [{"TEXT": {"REGEX": "((\d){5})"}}
                                                        ]
                }
            ]
#add patterns to ruler
ruler.add_patterns(patterns)


#create the doc
doc = nlp(text)

#extract entities
for ent in doc.ents:
    print (ent.text, ent.label_)