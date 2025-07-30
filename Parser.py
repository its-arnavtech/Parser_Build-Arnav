'''
Created on Monday, 21st August 2025
'''

import os
import spacy
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from pdfminer.high_level import extract_text
from docx import Document
from spacy.matcher import Matcher
from datetime import datetime
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
    education_keywords = [
        "bachelor", "master", "phd", "b.sc", "m.sc", "btech", "mtech",
        "mba", "b.e", "m.e", "bs", "ms", "university", "college", "institute",
        "degree", "school of", "graduated", "diploma", "high school"
    ]
    sents = text.split('\n')
    education = []

    for sent in sents:
        sent_lower = sent.lower()
        if any(keyword in sent_lower for keyword in education_keywords):
            sent_clean = re.sub(r'\s+', ' ', sent.strip())
            education.append(sent_clean)
    return education

def calculate_work_duration(start_date, end_date):
    try:
        start = datetime.strptime(start_date, '%b %Y')
        if end_date != "Present":
            end = datetime.strptime(end_date, '%b %Y')
        else:
            end = datetime.today()
        duration = end - start
        return duration.days // 30
    except Exception as e:
        print(f"Error getting duration: {e}")
        return None
        

def extract_work_experience(text):
    work_experiences = []
    pattern = r'(.+?)\s+at\s+(.+?),\s+([A-Za-z]+\s+\d{4})\s*-\s*([A-Za-z]+\s+\d{4}|Present)'
    matches = re.findall(pattern, text)
    for match in matches:
        job_title, company, start_date, end_date = match
        company = company.strip()
        start_date = start_date.strip()
        end_date = end_date.strip()

        duration = calculate_work_duration(start_date, end_date)
        if duration is not None:
            work_experiences.append({
                "company": company,
                "duration_months": duration,
                "start_date": start_date,
                "end_date": end_date
            })
    return work_experiences

def dump_to_json(data, filename="extracted_data.json"):
    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=3)
    print(f"Data dumped into {filename}")

def extract_data(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == '.docx':
        return extract_text_from_docx(file_path)
    else:
        print(f"Unsupported file type: {file_extension}")
        return None

file_paths = [
    'C:/Flexon_Resume_Parser/Parser_Build-Arnav/Resume_ArnavK.pdf',
    'C:/Flexon_Resume_Parser/Parser_Build-Arnav/ATS classic HR resume.docx'
]

result_data = {"pdf":{}, "docx":{}}

for file_path in file_paths:
    resume_text = extract_data(file_path)

    if resume_text:
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == ".pdf":
            result_section = result_data["pdf"]
        elif file_extension in ['.docx', '.doc']:
            result_section = result_data["docx"]
        else:
            print(f"Unsupported file format: {file_extension}")
            result_section = None

        if result_section is not None:
            names = extract_name(resume_text) or extract_name_regex(resume_text)
            email = extract_email(resume_text) or extract_email_regex(resume_text)
            phone_number = extract_phone_number_regex(resume_text) or extract_phone_number(resume_text)
            urls = extract_urls_spacy(resume_text) or []
            education = extract_education(resume_text) or []
            work_experiences = extract_work_experience(resume_text) or []

            result_section["names"] = names
            result_section["emails"] = email
            result_section["phone_numbers"] = phone_number
            result_section["urls"] = urls
            result_section["education"] = education
            result_section["work_experiences"] = work_experiences

        else:
            print(f"Error: Unsupported file type or failed to extract text from {file_path}.")
    else:
        print(f"Failed to extract text from {file_path}.")

dump_to_json(result_data)