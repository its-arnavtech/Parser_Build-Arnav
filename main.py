import os
import spacy
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from pdfminer.high_level import extract_text
from docx import Document
from spacy.matcher import Matcher
import json

#nltk.download('punkt')
#nltk.download('stopwords')

nlp = spacy.load("en_core_web_md")
global doc
ruler = nlp.add_pipe("entity_ruler", before="ner")
patterns = [
    {"label": "PHONE_NUMBER", "pattern": [{"ORTH": "("}, {"SHAPE": "ddd"}, {"ORTH": ")"}, {"SHAPE": "ddd"},
                                         {"ORTH": "-", "OP": "?"}, {"SHAPE": "dddd"}]},
    {"label": "PHONE_NUMBER", "pattern": [{"SHAPE": "ddd"}, {"SHAPE": "ddd"}, {"SHAPE": "dddd"}]}, 
]
ruler.add_patterns(patterns)

def extract_text_from_pdf(pdf_path):
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return None
    
def extract_text_from_docx(docx_path):
    try:
        doc = Document(docx_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
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
            names.append(ent.text)
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

def extract_education(text): #to fix
    keywords = ['education', 'degree', 'university', 'school', 'college']
    sents = sent_tokenize(text)
    education = []
    for sent in sents:
        sent_lower = sent.lower()
        found_keywords = []
        for keyword in keywords:
            if keyword in sent_lower:
                found_keywords.append(sent)

    return education

def dump_to_json(data, filename="extracted_data.json"):
    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=3)
    print(f"Data dumped into {filename}")

def extract_data(file_path):
    # Check the file extension to determine the type of document
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == '.docx':
        return extract_text_from_docx(file_path)
    else:
        print(f"Unsupported file type: {file_extension}")
        return None
'''
pdf_resume_text = extract_text_from_pdf('C:/Flexon_Resume_Parser/Parser_Build-Arnav/Resume_ArnavK.pdf')
doc_resume_text = extract_text_from_docx('C:/Flexon_Resume_Parser/Parser_Build-Arnav/ATS classic HR resume.docx')
'''
file_path = 'C:/Flexon_Resume_Parser/Parser_Build-Arnav/Resume_ArnavK.pdf'
resume_text = extract_data(file_path)
result_data = {"pdf":{}, "docx":{}}

if resume_text:
    #print(f"Extracting data from: {resume_text[:90]}...")
    names = extract_name(resume_text)
    if not names:
        names = extract_name_regex(resume_text)

    email = extract_email(resume_text)
    if not email:
        email = extract_email_regex(resume_text)

    phone_number = extract_phone_number_regex(resume_text)
    if not phone_number:
        phone_number = extract_phone_number(resume_text)
    
    urls = extract_urls_spacy(resume_text)
    if not urls:
        urls = []

    education = extract_education(resume_text)
    if not education:
        education = []

    # Assuming the file path corresponds to a PDF file
    result_data["pdf"]["names"] = names
    result_data["pdf"]["emails"] = email
    result_data["pdf"]["phone_numbers"] = phone_number
    result_data["pdf"]["urls"] = urls
    result_data["pdf"]["education"] = education

else:
    print("Failed to extract text from the Document.")

'''
if pdf_resume_text:
    #print(f"Extracting data from: {pdf_resume_text[:90]}...")
    names = extract_name(pdf_resume_text)
    if not names:
        names = extract_name_regex(pdf_resume_text)

    email = extract_email(pdf_resume_text)
    if not email:
        email = extract_email_regex(pdf_resume_text)

    phone_number = extract_phone_number_regex(pdf_resume_text)
    if not phone_number:
        phone_number = extract_phone_number(pdf_resume_text)
    
    urls = extract_urls_spacy(pdf_resume_text)
    if not urls:
        urls = []

    result_data["pdf"]["names"] = names
    result_data["pdf"]["emails"] = email
    result_data["pdf"]["phone_numbers"] = phone_number
    result_data["pdf"]["urls"] = urls

else:
    print("Failed to extract text from the PDF.")


if doc_resume_text:
    print(f"Extracting data from: {doc_resume_text[:90]}...")
    names = extract_name(doc_resume_text)
    if not names:
        names = extract_name_regex(doc_resume_text)

    email = extract_email(doc_resume_text)
    if not email:
        email = extract_email_regex(doc_resume_text)

    phone_number = extract_phone_number_regex(doc_resume_text)
    if not phone_number:
        phone_number = extract_phone_number(doc_resume_text)
    
    urls = extract_urls_spacy(doc_resume_text)
    if not urls:
        urls = []

    result_data["docx"]["names"] = names
    result_data["docx"]["emails"] = email
    result_data["docx"]["phone_numbers"] = phone_number
    result_data["docx"]["urls"] = urls

else:
    print("Failed to extract text from the Doc File.")
'''
dump_to_json(result_data)