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
global doc
ruler = nlp.add_pipe("entity_ruler", before="ner")
patterns = [
    {"label": "PHONE_NUMBER", "pattern": [{"ORTH": "("}, {"SHAPE": "ddd"}, {"ORTH": ")"}, {"SHAPE": "ddd"},
                                         {"ORTH": "-", "OP": "?"}, {"SHAPE": "dddd"}]},
    {"label": "PHONE_NUMBER", "pattern": [{"SHAPE": "ddd"}, {"SHAPE": "ddd"}, {"SHAPE": "dddd"}]},  # e.g., 7273948323
]
ruler.add_patterns(patterns)

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
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with single space

    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return filtered_tokens

def extract_name(text):
    lines = text.strip().split('\n')

    top_lines = ' '.join(lines[:5])
    doc = nlp(top_lines)
    names = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            names.append(ent)

    return names

def extract_name_regex(text):
    lines = text.strip().split('\n')
    top = ' '.join(lines[:5])
    pattern = r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b'
    match = re.findall(pattern, top)
    if match:
        return match[0]

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

def extract_phone_number(text):
    doc = nlp(text)
    phone_number = []
    for ent in doc.ents:
        if ent.label_ == "PHONE_NUMBER":
            phone_number.append(ent.text)
    return phone_number

def extract_phone_number_regex(text):
    phone_number_pattern = re.compile(r"(\+?\d{1,2}\s?)?(\(?\d{3}\)?[\s\-]?)?(\d{3})[\s\-]?(\d{4})")
    phone_numbers = []
    matches = phone_number_pattern.findall(text)
    for match in matches:
        phone_number = "".join(match).strip()
        if phone_number:
            phone_numbers.append(phone_number)
    return phone_numbers


def extract_urls_spacy(text):
    matcher = Matcher(nlp.vocab)
    pattern = [{"LIKE_URL": True}]
    matcher.add("URLS", [pattern])
    doc = nlp(text)
    matches = matcher(doc)

    urls = []
    for match_id, start, end in matches:
        span = doc[start:end]
        urls.append(span.text)

    return urls

sample_resume_text = extract_text_from_pdf('C:/Flexon_Resume_Parser/Parser_Build-Arnav/Resume_ArnavK.pdf')

if sample_resume_text:
    print(f"Extracting data from: {sample_resume_text[:90]}...")
    print("Extracting Name...")

    names = extract_name(sample_resume_text)
    if names:
        print(f"Name found: {names}")
    else:
        names = extract_name_regex(sample_resume_text)
        if names:
            print(f"Name found: {names}")
        else:
            print("No names found")

    print("Extracting Email ID...")

    email = extract_email(sample_resume_text)

    if email:
        print(f"Email(s) found using SpaCy: {email}")
    else:

        email = extract_email_regex(sample_resume_text)
        print(f"Email(s) found using regex: {email}")

    print("Extracting Phone Number...")

    phone_number = extract_phone_number_regex(sample_resume_text)
    if phone_number:
        print(f"Phone Number(s) found: {phone_number}")
    else:
        phone_number = extract_phone_number(sample_resume_text)
        print(f"Phone Number(s) found: {phone_number}")

    print("Extracting URL(s)...")
    urls = extract_urls_spacy(sample_resume_text)
    if urls:
        print(f"URL(s) found: {urls}")
    else:
        print("No URL(s) found.")

else:
    print("Failed to extract text from the PDF.")
'''

sample = "This is my ph nmber: (727)394-8323"
phNum = extract_phone_number(sample)
phNum2 = extract_phone_number_regex(sample)
print(phNum2)
'''