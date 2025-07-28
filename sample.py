import os
import nltk
from pdfminer.high_level import extract_text
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy

# Load the SpaCy model
nlp = spacy.load("en_core_web_md")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return None

# Path to the PDF file
pdf_resume_text = extract_text_from_pdf('C:/Flexon_Resume_Parser/Parser_Build-Arnav/Resume_ArnavK.pdf')

if pdf_resume_text:
    if not isinstance(pdf_resume_text, str):
        pdf_resume_text = str(pdf_resume_text)  # Ensure the text is a string
    
    # Tokenize the text into words
    tokens = word_tokenize(pdf_resume_text)

    # Print each token
    for token in tokens:
        print(token)
else:
    print("No text extracted from the PDF.")
