import os
import nltk
from pdfminer.high_level import extract_text
from nltk.tokenize import word_tokenize, sent_tokenize
from docx import Document
import spacy

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

def work_experience(text):
    doc = nlp(text)
    print("Organizations found:")
    for ent in doc.ents:
        if ent.label_ == "ORG":
            print(ent.text)

# Extract text from DOCX
doc_resume_text = extract_text_from_docx('C:/Flexon_Resume_Parser/Parser_Build-Arnav/ATS classic HR resume.docx')

# Run only if text was successfully extracted
if doc_resume_text:
    work_experience(doc_resume_text)
else:
    print("No text extracted from the resume.")
