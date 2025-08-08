'''
Created on Monday, 21st August 2025
Resume Parser - Optimized Production Version
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

# Enhanced patterns for phone number detection
ruler = nlp.add_pipe("entity_ruler", before="ner")
patterns = [
    {"label": "PHONE_NUMBER", "pattern": [{"ORTH": "("}, {"SHAPE": "ddd"}, {"ORTH": ")"}, {"SHAPE": "ddd"}, {"ORTH": "-", "OP": "?"}, {"SHAPE": "dddd"}]},
    {"label": "PHONE_NUMBER", "pattern": [{"SHAPE": "ddd"}, {"SHAPE": "ddd"}, {"SHAPE": "dddd"}]},
]
ruler.add_patterns(patterns)

def extract_text_from_pdf(pdf_path):
    try:
        return extract_text(pdf_path)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

def extract_text_from_docx(docx_path):
    try:
        doc = Document(docx_path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return None

def preprocessing_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    return [word for word in tokens if word.isalpha() and word not in stop_words]

def extract_name(text):
    lines = text.strip().split('\n')
    top_lines = ' '.join(lines[:20])
    
    # First try to find person names using spaCy
    doc = nlp(top_lines)
    potential_names = []
    
    for ent in doc.ents:
        if ent.label_ == "PERSON" and len(ent.text.split()) <= 3:
            # Filter out common non-names
            if not re.search(r'(?i)(university|college|institute|company|corporation|llc|inc)', ent.text):
                potential_names.append(ent.text)
    
    if potential_names:
        return potential_names[0]
    
    # Fallback: Look for capitalized names in first few lines
    for line in lines[:5]:
        line = line.strip()
        if not line or len(line) > 50:
            continue
        
        # Pattern for names (2-3 capitalized words)
        pattern = r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2}\b'
        matches = re.findall(pattern, line)
        
        for match in matches:
            # Filter out common non-names
            if not re.search(r'(?i)(university|college|institute|company|corporation|street|avenue|drive|resume|curriculum)', match):
                return match
    
    return None

def extract_email(text):
    doc = nlp(text)
    emails_spacy = [ent.text for ent in doc.ents if ent.label_ == "EMAIL"]
    if emails_spacy:
        return list(set(emails_spacy))
    
    # Enhanced regex for emails
    email_pattern = r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
    emails_regex = re.findall(email_pattern, text)
    return list(set(emails_regex))

def extract_phone_number(text):
    doc = nlp(text)
    phone_numbers = []
    
    # Enhanced regex patterns for phone numbers
    regex_patterns = [
        r'\+1[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        r'\+\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        r'\(\d{3}\)[-.\s]?\d{3}[-.\s]?\d{4}',
        r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b',
        r'\b\d{10}\b'
    ]
    
    for pattern in regex_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            # Clean and validate
            cleaned = re.sub(r'[^\d+]', '', match)
            digits_only = cleaned.lstrip('+1').lstrip('+')
            
            # Validation checks
            if (len(digits_only) == 10 and 
                not digits_only.startswith(('0', '1')) and
                len(set(digits_only)) > 3 and
                not re.match(r'\b(19|20)\d{8}\b', digits_only)):
                phone_numbers.append(match.strip())
    
    return list(set(phone_numbers))

def extract_urls_spacy(text):
    """Enhanced URL extraction with better filtering"""
    url_patterns = [
        r'https?://[^\s<>"\']+',
        r'www\.[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]*\.[a-zA-Z]{2,}(?:/[^\s]*)?',
        r'\b[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]*\.(?:com|org|net|edu|gov|mil)(?:/[^\s]*)?',
        r'linkedin\.com/in/[^\s]+',
        r'github\.com/[^\s]+'
    ]
    
    urls = []
    for pattern in url_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Ensure match is a string (handle tuple cases)
            if isinstance(match, tuple):
                match = match[0] if match else ''
            
            match = match.strip()
            if not match:
                continue
                
            # Filter out false positives - file extensions
            if re.search(r'(?i)\.(js|css|json|xml|pdf|doc|docx|txt|jpg|png|gif|py|java|cpp|c|h)(?:\s|$)', match):
                continue
                
            # Filter out programming/config file patterns
            if re.match(r'^[a-zA-Z0-9_-]+\.(js|css|json|py|java|cpp|c|h|xml)$', match):
                continue
                
            # Filter out patterns that are likely not URLs
            if len(match) < 5 or match.count('.') > 5:
                continue
                
            # Must contain valid URL characters
            if re.search(r'[a-zA-Z0-9]', match):
                urls.append(match)
    
    return list(set(urls))

def extract_education(text):
    education_info = []
    lines = text.split('\n')
    
    # Common degree patterns
    degree_patterns = [
        r'(?i)\b(bachelor|master|phd|doctorate|associate|diploma|certificate).*(?:degree|of|in)\b',
        r'(?i)\b(bs|ba|ms|ma|mba|phd|bsc|msc|beng|meng)\b',
        r'(?i)\b(b\.?\s*[sae]\.?|m\.?\s*[sae]\.?|ph\.?d\.?|m\.?b\.?a\.?)\b'
    ]
    
    # Institution patterns
    institution_patterns = [
        r'(?i)\b.+(?:university|college|institute|school)\b',
        r'(?i)\b(?:university|college|institute|school)\s+of\s+.+\b'
    ]
    
    for line in lines:
        line = line.strip()
        if not line or len(line) > 200:
            continue
        
        # Check for degrees
        for pattern in degree_patterns:
            if re.search(pattern, line):
                education_info.append(line)
                break
        
        # Check for institutions
        for pattern in institution_patterns:
            if re.search(pattern, line) and 'of' in line.lower():
                education_info.append(line)
                break
        
        # Check for GPA
        if re.search(r'(?i)\bgpa\b.*\d+\.\d+', line):
            education_info.append(line)
        
        # Check for academic honors
        if re.search(r'(?i)\b(dean.*list|honor|magna cum laude|summa cum laude|cum laude)\b', line):
            education_info.append(line)
    
    return list(set(education_info))

def extract_skills(text):
    skills = []
    lines = text.split('\n')
    in_skills_section = False
    
    # Skills section headers
    skills_headers = [
        r'(?i)^\s*(?:technical\s+)?skills?\s*$',
        r'(?i)^\s*core\s+competencies\s*$',
        r'(?i)^\s*programming\s+skills?\s*$',
        r'(?i)^\s*technologies\s*$'
    ]
    
    # Next section headers to stop at
    next_section_patterns = [
        r'(?i)^\s*(work\s+)?experience\s*$',
        r'(?i)^\s*education\s*$',
        r'(?i)^\s*projects?\s*$',
        r'(?i)^\s*certifications?\s*$',
        r'(?i)^\s*(professional\s+)?summary\s*$'
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if we're entering skills section
        for header_pattern in skills_headers:
            if re.match(header_pattern, line):
                in_skills_section = True
                continue
        
        # Check if we've left skills section
        if in_skills_section:
            for next_pattern in next_section_patterns:
                if re.match(next_pattern, line):
                    in_skills_section = False
                    break
        
        if in_skills_section:
            # Clean line of bullets and extract skills
            clean_line = re.sub(r'^[•\*\-\u25cf\u2022]\s*', '', line).strip()
            if clean_line and len(clean_line) > 2:
                # Split on common separators
                skill_items = re.split(r'[,;|•\u2022]', clean_line)
                for skill in skill_items:
                    skill = skill.strip()
                    if skill and 2 < len(skill) < 30:
                        skills.append(skill)
    
    # If no skills section found, look for comma-separated skill lists
    if not skills:
        for line in lines:
            if line.count(',') >= 3 and len(line) < 300:
                skill_items = line.split(',')
                if len(skill_items) >= 4:
                    for skill in skill_items:
                        skill = re.sub(r'^[•\*\-\u25cf\u2022]\s*', '', skill).strip()
                        if skill and 2 < len(skill) < 40:
                            skills.append(skill)
    
    return list(set(skills))

def total_experience(jobs):
    """Calculate total work experience in years"""
    if not jobs:
        return 0
    total_months = sum(job["duration_months"] for job in jobs)
    return round(total_months / 12, 2)

def extract_job_title_and_company(context_lines, date_line_index):
    """Enhanced job title and company extraction"""
    job_title = "Unknown Title"
    company = "Unknown Company"
    
    # Process context with spaCy for better entity recognition
    context_text = ' '.join(context_lines)
    doc = nlp(context_text)
    
    # Extract organizations
    organizations = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
    
    # Filter out common false positives for organizations
    filtered_orgs = []
    for org in organizations:
        if not re.search(r'(?i)(university|college|degree|bachelor|master|expected|graduation)', org):
            filtered_orgs.append(org)
    
    if filtered_orgs:
        company = filtered_orgs[0]
    
    # Look for job title in lines before the date
    for i in range(max(0, date_line_index - 3), date_line_index):
        if i < len(context_lines):
            line = context_lines[i].strip()
            
            # Skip lines that are likely not job titles
            if (not line or 
                len(line) > 100 or
                re.search(r'\d{4}', line) or
                re.search(r'(?i)(university|college|degree|bachelor|master|expected|graduation|texas|california|new york)', line) or
                line in organizations):
                continue
            
            # Clean the line
            clean_line = re.sub(r'^[•\*\-\u25cf\u2022]\s*', '', line).strip()
            
            # Check if it looks like a job title
            if (3 < len(clean_line) < 80 and 
                re.search(r'[A-Za-z]', clean_line) and
                not re.search(r'(?i)^(phone|email|address)', clean_line)):
                
                # Additional filtering for job titles
                if not re.search(r'(?i)(\.com|@|http|www)', clean_line):
                    job_title = clean_line
                    break
    
    return job_title, company

def calculate_work_duration(text):
    """Enhanced work experience extraction with better parsing"""
    date_patterns = [
        r'(?P<from>\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\s*(?:[-–—]|to)\s*(?P<to>(?:Present|Current|Now|Ongoing|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}))',
        r'(?P<from>\b\d{4})\s*(?:[-–—]|to)\s*(?P<to>(?:Present|Current|Now|Ongoing|\b\d{4}))',
        r'(?P<from>\b\d{1,2}/\d{4})\s*(?:[-–—]|to)\s*(?P<to>(?:Present|Current|Now|Ongoing|\b\d{1,2}/\d{4}))',
    ]

    jobs = []
    lines = text.split('\n')

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        for pattern in date_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                from_str = match.group('from')
                to_str = match.group('to')

                try:
                    start = dateparser.parse(from_str)
                    present_keywords = ['present', 'current', 'now', 'ongoing']
                    is_present = any(keyword in to_str.lower() for keyword in present_keywords)
                    
                    if is_present:
                        end = datetime.now()
                    else:
                        end = dateparser.parse(to_str)
                    
                    if not start or not end:
                        continue
                    
                    # Calculate duration
                    duration = relativedelta(end, start)
                    months = duration.years * 12 + duration.months
                    
                    # Add partial month for current jobs
                    if is_present:
                        days_in_month = end.day
                        month_fraction = days_in_month / 30
                        months += month_fraction
                    
                    if months <= 0:
                        continue

                    # Get context for job title and company extraction
                    context_start = max(0, i - 5)
                    context_end = min(len(lines), i + 3)
                    context_lines = []
                    
                    for j in range(context_start, context_end):
                        if lines[j].strip():
                            context_lines.append(lines[j].strip())
                    
                    # Extract job title and company
                    job_title, company = extract_job_title_and_company(context_lines, i - context_start)
                    
                    job_entry = {
                        "job_title": job_title,
                        "company": company,
                        "duration_months": round(months, 1),
                        "from": start.strftime('%b %Y'),
                        "to": end.strftime('%b %Y') if not is_present else "Present",
                        "is_current": is_present
                    }
                    
                    jobs.append(job_entry)
                    break
                    
                except Exception as e:
                    print(f"Error processing dates: {e}")
                    continue
    
    return jobs

def extract_work_experience(text):
    """Extract work experience section with better section detection"""
    section_headers = [
        r"(?i)^\s*(work\s+)?experience\s*$",
        r"(?i)^\s*professional\s+experience\s*$", 
        r"(?i)^\s*employment\s+(history|experience)\s*$",
        r"(?i)^\s*career\s+(history|experience)\s*$",
        r"(?i)^\s*work\s+history\s*$",
        r"(?i)^\s*professional\s+background\s*$"
    ]
    
    next_section_keywords = [
        r"(?i)^\s*education\s*$",
        r"(?i)^\s*(technical\s+)?skills\s*$",
        r"(?i)^\s*projects?\s*$",
        r"(?i)^\s*certifications?\s*$",
        r"(?i)^\s*(professional\s+)?summary\s*$",
        r"(?i)^\s*objective\s*$",
        r"(?i)^\s*achievements?\s*$",
        r"(?i)^\s*awards?\s*$",
        r"(?i)^\s*references?\s*$"
    ]
    
    lines = text.splitlines()
    experience_text = []
    recording = False
    section_found = False

    for line in lines:
        line_stripped = line.strip()
        
        # Check for experience section start
        for header_pattern in section_headers:
            if re.match(header_pattern, line_stripped):
                recording = True
                section_found = True
                break
        
        # Check for next section (stop recording)
        if recording:
            for next_pattern in next_section_keywords:
                if re.match(next_pattern, line_stripped):
                    recording = False
                    break
        
        # Collect experience text
        if recording and line_stripped:
            experience_text.append(line)
    
    result_text = "\n".join(experience_text)
    
    # If no explicit experience section found, return full text
    if not section_found:
        return text
    
    return result_text

def dump_to_json(data, filename="extracted_data.json"):
    try:
        with open(filename, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=3, ensure_ascii=False)
        print(f"Data extracted to {filename}")
        return True
    except Exception as e:
        print(f"Error saving to JSON: {e}")
        return False

def extract_data(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.docx':
        return extract_text_from_docx(file_path)
    else:
        print(f"Unsupported file type: {ext}")
        return None

# Main execution
if __name__ == "__main__":
    file_paths = [
        'C:/Flexon_Resume_Parser/Parser_Build-Arnav/Resume_ArnavK.pdf',
        'C:/Flexon_Resume_Parser/Parser_Build-Arnav/EPS-Computer-Science_sample.pdf',
        'C:/Flexon_Resume_Parser/Parser_Build-Arnav/ATS classic HR resume.docx'
    ]

    result_data = {"pdf": {}, "docx": {}}

    for file_path in file_paths:
        print(f"Processing: {file_path}")
        resume_text = extract_data(file_path)
        
        if not resume_text:
            print(f"Failed to extract text from {file_path}")
            continue

        file_extension = os.path.splitext(file_path)[1].lower()
        result_section = result_data["pdf"] if file_extension == ".pdf" else result_data["docx"]

        # Extract all information
        name = extract_name(resume_text)
        emails = extract_email(resume_text)
        phone_numbers = extract_phone_number(resume_text)
        urls = extract_urls_spacy(resume_text)
        education = extract_education(resume_text)
        skills = extract_skills(resume_text)

        # Extract work experience
        experience_section_text = extract_work_experience(resume_text)
        work_experiences = calculate_work_duration(experience_section_text) if experience_section_text else []
        total_exp_years = total_experience(work_experiences)

        file_data = {
            "name": name,
            "emails": emails,
            "phone_numbers": phone_numbers,
            "urls": urls,
            "education": education,
            "skills": skills,
            "work_experiences": work_experiences,
            "total_experience_years": total_exp_years
        }

        result_section[file_path] = file_data
        print(f"[OK] Processed {file_path}")

    # Save results
    dump_to_json(result_data)
print("[OK] All files processed successfully!")
