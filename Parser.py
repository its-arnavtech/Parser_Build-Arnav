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
        return None

def extract_text_from_docx(docx_path):
    try:
        doc = Document(docx_path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        return None

def preprocessing_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    return [word for word in tokens if word.isalpha() and word not in stop_words]

def extract_name(text):
    lines = text.strip().split('\n')
    top_lines = ' '.join(lines[:30])
    
    doc = nlp(top_lines)
    for ent in doc.ents:
        if ent.label_ == "PERSON" and len(ent.text.split()) <= 3:
            return ent.text
    pattern = r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b'
    matches = re.findall(pattern, top_lines)
    return matches[0] if matches else None

def extract_email(text):
    doc = nlp(text)
    emails_spacy = [ent.text for ent in doc.ents if ent.label_ == "EMAIL"]
    if emails_spacy:
        return list(set(emails_spacy))
    
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails_regex = re.findall(email_pattern, text)
    return list(set(emails_regex))

def extract_phone_number(text):
    doc = nlp(text)
    matcher = Matcher(nlp.vocab)
    phone_pattern = [
        {"TEXT": {"REGEX": r"[\(\[]?"}},
        {"TEXT": {"REGEX": r"\d{3}"}},
        {"TEXT": {"REGEX": r"[\)\]]?"}},
        {"TEXT": {"REGEX": r"[-.\s]?"}},
        {"TEXT": {"REGEX": r"\d{3}"}},
        {"TEXT": {"REGEX": r"[-.\s]?"}},
        {"TEXT": {"REGEX": r"\d{4}"}}
    ]
    matcher.add("PHONE_PATTERN", [phone_pattern])
    matches = matcher(doc)
    matched_spans = [doc[start:end] for _, start, end in matches]
    with doc.retokenize() as retokenizer:
        for span in matched_spans:
            retokenizer.merge(span)
    chunked_doc = nlp(doc.text)
    phone_numbers = []
    for ent in chunked_doc.ents:
        if ent.label_.lower() == "phone_number":
            phone_numbers.append(ent.text.strip())

    if not phone_numbers:
        regex_patterns = [
            r'\+\d{1,3}[\s\-\.]?\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4}',
            r'\(\d{3}\)[\s\-\.]?\d{3}[\s\-\.]?\d{4}',
            r'\b\d{3}[\s\-\.]\d{3}[\s\-\.]\d{4}\b',
            r'\b\d{10}\b'
        ]
        for pattern in regex_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                cleaned = re.sub(r'[^\d+]', '', match)
                digits_only = cleaned.lstrip('+1').lstrip('+')
                if (len(digits_only) == 10 and 
                    not digits_only.startswith(('19', '20')) and
                    not re.search(r'\b(19|20)\d{2}[\s\-](19|20)\d{2}\b', match) and
                    len(set(digits_only)) > 1 and
                    digits_only[0] not in ['0', '1']):
                    phone_numbers.append(match.strip())

    return list(set(phone_numbers))

def extract_urls_spacy(text):
    matcher = Matcher(nlp.vocab)
    matcher.add("URLS", [[{"LIKE_URL": True}]])
    doc = nlp(text)
    matches = matcher(doc)
    return [doc[start:end].text for _, start, end in matches]

def extract_education(text):
    education_info = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if re.search(r'(?i)\buniversity\b|\bcollege\b|\binstitute\b', line) and 'of' in line.lower():
            education_info.append(line)
            
        if re.search(r'(?i)\b(bs|ba|ms|ma|bachelor|master|phd|mba)\b.*\(?\b(19|20)\d{2}\)?\b', line):
            education_info.append(line)
            
        if re.search(r'(?i)\bgpa\b', line):
            education_info.append(line)
            
        if re.search(r'(?i)\bdean.*list\b|\bchancellor.*list\b', line):
            education_info.append(line)
    
    return education_info

def extract_skills(text):
    skills = []
    lines = text.split('\n')
    in_skills_section = False
    
    section_headers = [
        'work experience', 'professional experience', 'experience', 'education', 
        'projects', 'certifications', 'summary', 'objective', 'achievements',
        'awards', 'contact', 'personal information', 'references'
    ]
    
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
            clean_line = re.sub(r'^[•\*\-\u25cf]\s*', '', line).strip()
            if clean_line and len(clean_line) > 2:
                skill_items = re.split(r'[,;|•]', clean_line)
                for skill in skill_items:
                    skill = skill.strip()
                    if (skill and len(skill) > 2 and len(skill) < 50 and
                        skill.lower() not in section_headers and
                        not re.search(r'(?i)^(intern|engineer|developer|analyst|manager|specialist)$', skill) and
                        not skill.endswith('.') and
                        not re.search(r'(?i)(team|job|process|based access control)', skill)):
                        skills.append(skill)
                        
    if not skills:
        for line in lines:
            if line.count(',') >= 2 and len(line) < 200:
                skill_items = line.split(',')
                if len(skill_items) >= 3:
                    for skill in skill_items:
                        skill = re.sub(r'^[•\*\-\u25cf]\s*', '', skill).strip()
                        if (skill and len(skill) > 2 and len(skill) < 30 and
                            skill.lower() not in section_headers and
                            not re.search(r'(?i)^(intern|engineer|developer|analyst|manager|specialist)$', skill) and
                            not skill.endswith('.') and
                            not re.search(r'(?i)(team|job|process|based access control)', skill)):
                            skills.append(skill)
    
    return list(set(skills))

def extract_work_description(text, job_title, company, start_line_idx, lines):
    description_lines = []
    
    for i in range(start_line_idx + 1, min(len(lines), start_line_idx + 20)):
        if i >= len(lines):
            break
            
        line = lines[i].strip()
        
        if re.search(r'(?P<from>\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Sept|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\s*(?:--|[-–—to]|to)\s*(?P<to>(?:Present|Current|Now|Ongoing|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Sept|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}))', line, re.IGNORECASE):
            break
            
        if re.search(r'(?i)^\s*(##|education|skills|projects|certifications|summary|objective|achievements|awards)\s*$', line):
            break
            
        if not line:
            continue
            
        clean_line = re.sub(r'^[•\*\-\u25cf➤→]\s*', '', line).strip()
        clean_line = re.sub(r'^\d+\.\s*', '', clean_line).strip()
        
        if (clean_line and 
            len(clean_line) > 15 and 
            len(clean_line) < 500 and
            not re.search(r'(?i)^(contact|phone|email|address)', clean_line) and
            not re.search(r'^[A-Z][a-z]+\s+[A-Z][a-z]+$', clean_line) and
            not re.search(r'(?i)^(company|organization|location)', clean_line)):
                description_lines.append(clean_line)
    
    return ' '.join(description_lines) if description_lines else "No description available"

def total_experience(jobs):
    if not jobs:
        return 0
    total_months = sum(job["duration_months"] for job in jobs)
    return round(total_months / 12, 2)

def calculate_work_duration(text):
    if not text:
        return []
    
    date_patterns = [
        r'(?P<from>\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Sept|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\s*(?:--|[-–—]|to)\s*(?P<to>(?:Present|Current|Now|Ongoing|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Sept|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}))',
        r'(?P<from>\b\d{4})\s*(?:--|[-–—]|to)\s*(?P<to>(?:Present|Current|Now|Ongoing|\b\d{4}))',
        r'(?P<from>\b\d{1,2}/\d{4})\s*(?:--|[-–—]|to)\s*(?P<to>(?:Present|Current|Now|Ongoing|\b\d{1,2}/\d{4}))',
    ]

    jobs = []
    lines = text.split('\n')

    for i, line in enumerate(lines):
        original_line = line
        line = line.strip()
        if not line:
            continue
            
        for pattern in date_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                from_str = match.group('from')
                to_str = match.group('to')

                try:
                    from_str_normalized = from_str.replace('Sept', 'Sep')
                    start = dateparser.parse(from_str_normalized)
                    
                    present_keywords = ['present', 'current', 'now', 'ongoing']
                    if any(keyword in to_str.lower() for keyword in present_keywords):
                        today = datetime.now()
                        if today.month == 12:
                            end = datetime(today.year + 1, 1, 1) - relativedelta(days=1)
                        else:
                            end = datetime(today.year, today.month + 1, 1) - relativedelta(days=1)
                    else:
                        to_str_normalized = to_str.replace('Sept', 'Sep')
                        end = dateparser.parse(to_str_normalized)
                    
                    if not start or not end:
                        continue
                        
                    duration = relativedelta(end, start)
                    months = duration.years * 12 + duration.months
                    
                    if any(keyword in to_str.lower() for keyword in present_keywords):
                        if start.day > 1:
                            days_in_current_month = (end.replace(day=1) + relativedelta(months=1) - relativedelta(days=1)).day
                            current_day = datetime.now().day
                            month_fraction = current_day / days_in_current_month
                            months += month_fraction
                    
                    if months <= 0:
                        continue

                    job_title = "Unknown Title"
                    company = "Unknown Company"
                    
                    # Look for job title in previous lines (markdown headers or job titles)
                    for j in range(max(0, i-5), i):
                        if j < len(lines):
                            candidate_line = lines[j].strip()
                            
                            # Check for markdown headers (##)
                            if candidate_line.startswith('##'):
                                title_candidate = candidate_line.replace('##', '').strip()
                                if title_candidate and len(title_candidate) < 80:
                                    job_title = title_candidate
                                    break
                            
                            # Check for lines with job keywords
                            job_keywords = ['engineer', 'scientist', 'analyst', 'manager', 'developer', 'specialist']
                            if any(keyword.lower() in candidate_line.lower() for keyword in job_keywords):
                                if len(candidate_line) < 80 and not re.search(r'\d{4}', candidate_line):
                                    job_title = candidate_line
                                    break
                    
                    # Look for company using NER and context
                    context_lines = []
                    for j in range(max(0, i-3), min(len(lines), i+4)):
                        if lines[j].strip():
                            context_lines.append(lines[j])
                    
                    context = ' '.join(context_lines)
                    doc = nlp(context)
                    orgs = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
                    
                    if orgs:
                        company = orgs[0]
                    else:
                        # Look for company in lines after job title
                        for j in range(i+1, min(len(lines), i+4)):
                            if j < len(lines):
                                candidate_line = lines[j].strip()
                                if (candidate_line and 
                                    not re.search(r'[-•]', candidate_line) and
                                    not candidate_line.lower().startswith(('integrated', 'developed', 'implemented', 'built')) and
                                    len(candidate_line.split()) <= 4 and
                                    len(candidate_line) > 3):
                                    company = candidate_line.replace(',', '').replace('.', '').strip()
                                    break

                    work_description = extract_work_description(text, job_title, company, i, lines)

                    job_entry = {
                        "job_title": job_title,
                        "company": company,
                        "duration_months": round(months, 1),
                        "from": start.strftime('%b %Y'),
                        "to": end.strftime('%b %Y') if not any(keyword in to_str.lower() for keyword in present_keywords) else "Present",
                        "is_current": any(keyword in to_str.lower() for keyword in present_keywords),
                        "work_description": work_description
                    }
                    
                    jobs.append(job_entry)
                    break
                    
                except Exception:
                    continue
                    
    return jobs

def extract_work_experience(text):
    section_headers = [
        r"(?i)work[\s\*\[\]]*experience",
        r"(?i)professional[\s\*\[\]]*experience", 
        r"(?i)employment[\s\*\[\]]*history",
        r"(?i)career[\s\*\[\]]*history",
        r"(?i)experience[\s\*\[\]]*:",
        r"(?i)\*\*.*work.*experience.*\*\*",
        r"(?i)\[work.*experience.*\]"
    ]
    
    next_section_keywords = [
        r"(?i)education",
        r"(?i)technical[\s\*\[\]]*skills",
        r"(?i)skills[\s\*\[\]]*:",
        r"(?i)projects",
        r"(?i)certifications?",
        r"(?i)summary",
        r"(?i)objective",
        r"(?i)achievements?",
        r"(?i)awards?",
        r"(?i)\*\*.*education.*\*\*"
    ]
    
    lines = text.splitlines()
    experience_text = []
    recording = False
    section_found = False
    
    for i, line in enumerate(lines):
        original_line = line
        line_stripped = line.strip()
        
        if not recording:
            for header_pattern in section_headers:
                if re.search(header_pattern, line_stripped):
                    recording = True
                    section_found = True
                    break
        
        if recording:
            section_ended = False
            for next_pattern in next_section_keywords:
                if re.search(next_pattern, line_stripped):
                    recording = False
                    section_ended = True
                    break
            
            if recording and not section_ended:
                if original_line.strip():
                    experience_text.append(original_line)
    
    return '\n'.join(experience_text) if experience_text else None

def split_into_sections(text):
    section_titles = {
        "education": ["education", "academic", "background", "qualifications"],
        "experience": ["experience", "work", "professional", "employment", "career"],
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
    try:
        with open(filename, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=3, ensure_ascii=False)
        print(f"Data extracted to {filename}")
        return True
    except Exception as e:
        return False

def extract_data(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.docx':
        return extract_text_from_docx(file_path)
    else:
        return None

file_paths = [
    'C:/Flexon_Resume_Parser/Parser_Build-Arnav/GIRISH GUPTA.docx'
]

result_data = {"pdf": {}, "docx": {}}

for file_path in file_paths:
    resume_text = extract_data(file_path)
    
    if not resume_text:
        continue

    file_extension = os.path.splitext(file_path)[1].lower()
    result_section = result_data["pdf"] if file_extension == ".pdf" else result_data["docx"]

    name = extract_name(resume_text)
    emails = extract_email(resume_text)
    phone_numbers = extract_phone_number(resume_text)
    urls = extract_urls_spacy(resume_text)
    education = extract_education(resume_text)
    skills = extract_skills(resume_text)
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

dump_to_json(result_data)