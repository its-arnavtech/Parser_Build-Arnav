import spacy
import nltk
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk

nlp = spacy.load("en_core_web_md")

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')
nltk.download('averaged_perception_tagger')
nltk.download('maxent_ne_chunker')
