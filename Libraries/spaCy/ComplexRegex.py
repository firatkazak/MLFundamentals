import re

text = "Paul Newman was an American actor, but Paul Hollywood is a British TV Host. The name Paul is quite common."

pattern = r"Paul [A-Z]\w+"

matches = re.finditer(pattern, text)

for match in matches:
    print(match)

import re
import spacy
from spacy.tokens import Span

text = "Paul Newman was an American actor, but Paul Hollywood is a British TV Host. The name Paul is quite common."
pattern = r"Paul [A-Z]\w+"

nlp = spacy.blank("en")
doc = nlp(text)

original_ents = list(doc.ents)

mwt_ents = []
for match in re.finditer(pattern, doc.text):
    start, end = match.span()
    span = doc.char_span(start, end)
    if span is not None:
        mwt_ents.append((span.start, span.end, span.text))

for ent in mwt_ents:
    start, end, name = ent
    per_ent = Span(doc, start, end, label="PERSON")
    original_ents.append(per_ent)

doc.ents = original_ents

for ent in doc.ents:
    print(ent.text, ent.label_)

import re
import spacy

text = "Paul Newman was an American actor, but Paul Hollywood is a British TV Host."
pattern = r"Hollywood"

nlp = spacy.load("en_core_web_sm")

doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.label_)

from spacy.util import filter_spans

filtered = filter_spans(original_ents)
doc.ents = filtered
for ent in doc.ents:
    print(ent.text, ent.label_)
