import spacy
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from pdfminer.high_level import extract_text
#import sys
from spacy.matcher import Matcher

sample_resume_text = extract_text(r'C:\Flexon_Resume_Parser\Parser_Build-Arnav\Resume_ArnavK.pdf').encode('utf-8')

nlp = spacy.load("en_core_web_md")

#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('words')
#nltk.download('averaged_perception_tagger')
#nltk.download('maxent_ne_chunker')

def preprocessing_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower() #lowercase to avoid issues and treat all words equal
    tokens = word_tokenize(text) #tokenization
    stop_words = set(stopwords.words("english"))

    filtered_tokens = []
    for word in tokens:
        if word.isalpha() and word not in stop_words:
            filtered_tokens.append(word)

    return filtered_tokens

#filtered_tokens = preprocessing_text(sample_resume_text)
#print(filtered_tokens)

#def extract_name:

def extract_email(text):
    doc = nlp(text)
    emails = []

    for ent in doc.ents:
        if ent.label_ == "EMAIL":
            emails.append(ent.text)

    print(emails)

extract_email(sample_resume_text)