import os
import nltk
from pdfminer.high_level import extract_text
from nltk.tokenize import word_tokenize, sent_tokenize
from docx import Document
import spacy
import re
import dateparser
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Load the SpaCy model
nlp = spacy.load("en_core_web_md")

def extract_text_from_docx(doc_path):
    try:
        doc = Document(doc_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return None

# ✅ NEW FUNCTION: Extract only the Experience section
def extract_experience_section(text):
    section_headers = [
        "experience", "work experience", "professional experience",
        "employment history", "career summary"
    ]
    next_section_keywords = [
        "education", "skills", "projects", "certifications", "summary", "objective"
    ]

    lines = text.splitlines()
    experience_text = []
    recording = False

    for line in lines:
        clean_line = line.strip().lower()

        if any(h in clean_line for h in section_headers):
            recording = True
            continue

        if recording and any(k in clean_line for k in next_section_keywords):
            break

        if recording:
            experience_text.append(line)

    return "\n".join(experience_text)

# Re-load SpaCy smaller model (if needed)
nltk.download('punkt')
nlp = spacy.load("en_core_web_md")

def extract_date_ranges(text):
    date_pattern = r'(?P<from>\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|' \
                   r'Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})' \
                   r'\s*(?:to|–|-)\s*' \
                   r'(?P<to>(?:Present|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|' \
                   r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}))'

    matches = re.findall(date_pattern, text, re.IGNORECASE)
    date_ranges = []
    for start_str, end_str in matches:
        start = dateparser.parse(start_str)
        end = dateparser.parse(end_str) if end_str.lower() != 'present' else datetime.now()
        if start and end:
            date_ranges.append((start, end))
    return date_ranges

def calculate_total_experience(date_ranges):
    total_months = 0
    for start, end in date_ranges:
        delta = relativedelta(end, start)
        months = delta.years * 12 + delta.months
        total_months += months
    return round(total_months / 12, 2)

def extract_titles_and_companies(text):
    titles = []
    companies = []
    for sent in nltk.sent_tokenize(text):
        doc = nlp(sent)
        for ent in doc.ents:
            if ent.label_ == "ORG":
                companies.append(ent.text)
            elif ent.label_ in ["PERSON", "WORK_OF_ART", "TITLE"]:
                titles.append(ent.text)
        if " at " in sent.lower():
            parts = sent.split(" at ")
            titles.append(parts[0].strip())
            companies.append(parts[1].strip())
    return list(set(titles)), list(set(companies))

def extract_experience_info(text):
    date_ranges = extract_date_ranges(text)
    total_exp = calculate_total_experience(date_ranges)
    titles, companies = extract_titles_and_companies(text)

    return {
        "total_experience_years": total_exp,
        "job_titles": titles,
        "company_names": companies,
        "experience_date_ranges": date_ranges
    }

# Main execution
if __name__ == "__main__":
    doc_resume_text = extract_text_from_docx('C:/Flexon_Resume_Parser/Parser_Build-Arnav/ATS classic HR resume.docx')

    if doc_resume_text:
        experience_section = extract_experience_section(doc_resume_text)

        if experience_section.strip():
            experience_data = extract_experience_info(experience_section)
            from pprint import pprint
            pprint(experience_data)
        else:
            print("Could not find an Experience section in the resume.")
    else:
        print("No text extracted from the resume.")
