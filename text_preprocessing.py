import spacy
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pdfminer.high_level import extract_text
from spacy.matcher import Matcher

#nltk.download('punkt')
#nltk.download('stopwords')

nlp = spacy.load("en_core_web_md")

def extract_text_from_pdf(pdf_path):
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return None

def preprocessing_text(text):
    if not isinstance(text, str):
        text = str(text)
    tokens = word_tokenize(text)
    text = text.lower()
    stop_words = set(stopwords.words("english"))

    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return filtered_tokens

def extract_name(text):
    ...

def extract_email(text):
    if not text:
        print("No data extracted")
        return []

    doc = nlp(text)
    emails = [ent.text for ent in doc.ents if ent.label_ == "EMAIL"]
    return emails

def extract_email_regex(text):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return re.findall(email_pattern, text)

sample_resume_text = extract_text_from_pdf(r'C:\Flexon_Resume_Parser\Parser_Build-Arnav\Resume_ArnavK.pdf')

if sample_resume_text:
    print(f"Extracting data from: {sample_resume_text[:100]}...")
    print("Extracting Email ID...")

    email = extract_email(sample_resume_text)

    if email:
        print(f"Email(s) found using SpaCy: {email}")
    else:

        email = extract_email_regex(sample_resume_text)
        print(f"Email(s) found using regex: {email}")
else:
    print("Failed to extract text from the PDF.")
