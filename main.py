import spacy

nlp = spacy.blank("en")
nlp.add_pipe('sentencizer')
