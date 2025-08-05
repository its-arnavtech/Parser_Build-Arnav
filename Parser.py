'''
Created on Monday, 21st August 2025
'''

import os
import spacy
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pdfminer.high_level import extract_text
from docx import Document
from spacy.matcher import Matcher
from datetime import datetime
from dateutil.relativedelta import relativedelta
import dateparser
import json

nlp = spacy.load("en_core_web_md")

ruler = nlp.add_pipe("entity_ruler", before="ner")
patterns = [
    {"label": "PHONE_NUMBER", "pattern": [{"ORTH": "("}, {"SHAPE": "ddd"}, {"ORTH": ")"}, {"SHAPE": "ddd"},
                                         {"ORTH": "-", "OP": "?"}, {"SHAPE": "dddd"}]},
    {"label": "PHONE_NUMBER", "pattern": [{"SHAPE": "ddd"}, {"SHAPE": "ddd"}, {"SHAPE": "dddd"}]},
]
ruler.add_patterns(patterns)

# nltk.download('punkt')
# nltk.download('stopwords')

def extract_text_from_pdf(pdf_path):
    try:
        return extract_text(pdf_path)
    except Exception as e:
        print(f"Error extracting text: {e}")
        return None

def extract_text_from_docx(docx_path):
    try:
        doc = Document(docx_path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        print(f"Error extracting text: {e}")
        return None

def preprocessing_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    return [word for word in tokens if word.isalpha() and word not in stop_words]

def extract_name(text):
    lines = text.strip().split('\n')
    top_lines = ' '.join(lines[:30])
    
    preprocessed_top = preprocessing_text(top_lines)
    
    doc = nlp(top_lines)
    for ent in doc.ents:
        if ent.label_ == "PERSON" and len(ent.text.split()) <= 3:
            return ent.text
    
    pattern = r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b'
    matches = re.findall(pattern, top_lines)
    return matches[0] if matches else None

def extract_email(text):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, text)
    return list(set(emails))

def extract_phone_number(text):
    phone_patterns = [
        # International format
        r'\+\d{1,3}[\s\-\.]?\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4}',
        # US format with parentheses: (123) 456-7890
        r'\(\d{3}\)[\s\-\.]?\d{3}[\s\-\.]?\d{4}',
        # Standard formats: 123-456-7890, 123.456.7890, 123 456 7890
        r'\b\d{3}[\s\-\.]\d{3}[\s\-\.]\d{4}\b',
        # 10 digits together: 1234567890 (but only if it looks like a phone number)
        r'\b\d{10}\b'
    ]
    phone_numbers = []
    for pattern in phone_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            cleaned = re.sub(r'[^\d+]', '', match)
            digits_only = cleaned.lstrip('+1').lstrip('+')
            if len(digits_only) != 10:
                continue
            if digits_only.startswith(('19', '20')) and len(digits_only) == 4:
                continue
            year_pattern = r'\b(19|20)\d{2}[\s\-](19|20)\d{2}\b'
            if re.search(year_pattern, match):
                continue
            if len(set(digits_only)) == 1:
                continue
            if digits_only[0] in ['0', '1']:
                continue
            phone_numbers.append(match.strip())
    return list(set(phone_numbers))

def extract_urls_spacy(text):
    matcher = Matcher(nlp.vocab)
    matcher.add("URLS", [[{"LIKE_URL": True}]])
    doc = nlp(text)
    matches = matcher(doc)
    return [doc[start:end].text for _, start, end in matches]

def extract_education(text):
    """Extract education information - simple and effective approach"""
    education_info = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        #universities/colleges
        if re.search(r'(?i)\buniversity\b|\bcollege\b|\binstitute\b', line) and 'of' in line.lower():
            education_info.append(line)
            
        #degrees with years
        if re.search(r'(?i)\b(bs|ba|ms|ma|bachelor|master|phd|mba)\b.*\(?\b(19|20)\d{2}\)?\b', line):
            education_info.append(line)
            
        #GPA
        if re.search(r'(?i)\bgpa\b', line):
            education_info.append(line)
            
        #Dean's list etc
        if re.search(r'(?i)\bdean.*list\b|\bchancellor.*list\b', line):
            education_info.append(line)
    
    return education_info

def extract_skills(text):
    """Extract skills information"""
    skills = []
    lines = text.split('\n')
    in_skills_section = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.search(r'(?i)^skills?$|^technical skills?$|^core competencies$', line):
            in_skills_section = True
            continue
        if in_skills_section and re.search(r'(?i)^(experience|education|projects|certifications|summary|objective)', line):
            break
        if in_skills_section:
            clean_line = re.sub(r'^[•\*\-]\s*', '', line).strip()
            if clean_line and len(clean_line) > 2:
                skill_items = re.split(r'[,;|•]', clean_line)
                for skill in skill_items:
                    skill = skill.strip()
                    if skill and len(skill) > 2 and len(skill) < 50:
                        skills.append(skill)
    
    if not skills:
        for line in lines:
            if line.count(',') >= 2 and len(line) < 200:
                skill_items = line.split(',')
                if len(skill_items) >= 3:
                    for skill in skill_items:
                        skill = re.sub(r'^[•\*\-]\s*', '', skill).strip()
                        if skill and len(skill) > 2 and len(skill) < 30:
                            skills.append(skill)
    
    return list(set(skills))

def total_experience(jobs):
    total_months = sum(job["duration_months"] for job in jobs)
    return round(total_months / 12, 2)

def calculate_work_duration(text):
    date_pattern = r'(?P<from>\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|' \
                   r'Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})' \
                   r'\s*(?:to|–|-)\s*' \
                   r'(?P<to>(?:Present|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|' \
                   r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}))'

    jobs = []
    lines = text.split('\n')

    for i, line in enumerate(lines):
        match = re.search(date_pattern, line, re.IGNORECASE)
        if match:
            from_str = match.group('from')
            to_str = match.group('to')

            start = dateparser.parse(from_str)
            end = dateparser.parse(to_str) if 'present' not in to_str.lower() else datetime.now()
            if not start or not end:
                continue

            months = relativedelta(end, start).years * 12 + relativedelta(end, start).months

            context = ' '.join(lines[max(0, i - 2): i + 2]).strip()
            doc = nlp(context)
            orgs = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
            titles = [ent.text for ent in doc.ents if ent.label_ in ['TITLE', 'WORK_OF_ART']]

            jobs.append({
                "job_title": titles[0] if titles else "Unknown Title",
                "company": orgs[0] if orgs else "Unknown Company",
                "duration_months": months,
                "from": start.strftime('%b %Y'),
                "to": end.strftime('%b %Y') if 'present' not in to_str.lower() else "Present"
            })
    return jobs

def extract_work_experience(text):
    section_headers = [
        "experience", "work", "professional", "employment", "history", "career", "summary"
    ]
    next_section_keywords = [
        "education", "skills", "projects", "certifications", "summary", "objective"
    ]
    
    lines = text.splitlines()
    experience_text = []
    recording = False

    for line in lines:
        preprocessed_line = preprocessing_text(line)
        
        if any(header in preprocessed_line for header in section_headers):
            recording = True
            continue
        if recording and any(keyword in preprocessed_line for keyword in next_section_keywords):
            break
            
        if recording:
            experience_text.append(line)

    return "\n".join(experience_text)

def split_into_sections(text):
    section_titles = {
        "education": ["education", "academic", "background", "qualifications"],
        "experience": ["experience", "work", "professional", "employment"],
        "skills": ["skills", "technical", "core", "competencies"],
        "projects": ["projects", "personal"],
        "certifications": ["certifications", "licenses", "certificates"],
        "summary": ["summary", "profile", "objective"],
    }

    sections = {}
    current_section = None
    buffer = []
    
    for line in text.splitlines():
        preprocessed_line = preprocessing_text(line)
        found = False
        
        for key, keywords in section_titles.items():
            if any(keyword in preprocessed_line for keyword in keywords):
                if current_section and buffer:
                    sections[current_section] = '\n'.join(buffer).strip()
                current_section = key
                buffer = []
                found = True
                break
                
        if not found and current_section:
            buffer.append(line)
            
    if current_section and buffer:
        sections[current_section] = '\n'.join(buffer).strip()
        
    return sections

def dump_to_json(data, filename="extracted_data.json"):
    with open(filename, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=3, ensure_ascii=False)
    print(f"Data saved to {filename}")

def extract_data(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.docx':
        return extract_text_from_docx(file_path)
    else:
        print(f"❌ Unsupported file: {ext}")
        return None

file_paths = [
    'C:/Flexon_Resume_Parser/Parser_Build-Arnav/Resume_ArnavK.pdf',
    'C:/Flexon_Resume_Parser/Parser_Build-Arnav/functionalsample.pdf',
    'C:/Flexon_Resume_Parser/Parser_Build-Arnav/ATS classic HR resume.docx'
]

result_data = {"pdf": {}, "docx": {}}

for file_path in file_paths:
    resume_text = extract_data(file_path)
    if not resume_text:
        print(f"❌ Could not extract text from: {file_path}")
        continue

    sections = split_into_sections(resume_text)
    file_extension = os.path.splitext(file_path)[1].lower()
    result_section = result_data["pdf"] if file_extension == ".pdf" else result_data["docx"]

    name = extract_name(resume_text)
    emails = extract_email(resume_text)
    phone_numbers = extract_phone_number(resume_text)
    urls = extract_urls_spacy(resume_text)
    education = extract_education(resume_text)
    skills = extract_skills(resume_text)
    
    experience_section_text = extract_work_experience(sections.get("experience", ""))
    work_experiences = calculate_work_duration(experience_section_text) if experience_section_text else []
    total_exp_years = total_experience(work_experiences)

    result_section[file_path] = {
        "name": name,
        "emails": emails,
        "phone_numbers": phone_numbers,
        "urls": urls,
        "education": education,
        "skills": skills,
        "work_experiences": work_experiences,
        "total_experience_years": total_exp_years
    }

dump_to_json(result_data)