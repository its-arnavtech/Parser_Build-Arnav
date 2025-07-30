'''This file is only to see what data the pdf miner extracts from the pdf file'''

import os
import nltk
from pdfminer.high_level import extract_text
from nltk.tokenize import word_tokenize, sent_tokenize
from docx import Document
import spacy

# Load the SpaCy model
nlp = spacy.load("en_core_web_md")
docx = 'C:/Flexon_Resume_Parser/Parser_Build-Arnav/ATS classic HR resume.docx'
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


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return None

# Path to the PDF file

doc_resume_text = extract_text_from_docx('C:/Flexon_Resume_Parser/Parser_Build-Arnav/ATS classic HR resume.docx')
if doc_resume_text:
    if not isinstance(doc_resume_text, str):
        pdf_resume_text = str(doc_resume_text)  # Ensure the text is a string
    
    # Tokenize the text into words
    tokens = word_tokenize(doc_resume_text)

    # Print each token
    for token in tokens:
        print(token)
else:
    print("No text extracted from the PDF.")
