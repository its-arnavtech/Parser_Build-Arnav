import os
import spacy
import re
import json
from pdfminer.high_level import extract_text
from docx import Document
from datetime import datetime
from dateutil.relativedelta import relativedelta
import dateparser

# Load spaCy model
nlp = spacy.load("en_core_web_md")

def extract_text_from_pdf(pdf_path):
    try:
        return extract_text(pdf_path)
    except Exception:
        return None

def extract_text_from_docx(docx_path):
    try:
        doc = Document(docx_path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return None

def extract_name(text):
    lines = text.strip().split('\n')[:5]  # Check first 5 lines
    
    # Try spaCy NER first
    doc = nlp(' '.join(lines))
    for ent in doc.ents:
        if ent.label_ == "PERSON" and len(ent.text.split()) <= 3:
            return ent.text.strip()
    
    # Fallback to regex for capitalized names
    for line in lines:
        line = line.strip()
        # Match 2-3 capitalized words at start of line
        match = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})(?:\s|$)', line)
        if match and not re.search(r'\d|@|\.com', match.group(1)):
            return match.group(1)
    
    return None

def extract_email(text):
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(pattern, text)
    return list(set(emails))

def extract_phone_number(text):
    patterns = [
        r'\+?1?[-\.\s]?\(?[0-9]{3}\)?[-\.\s]?[0-9]{3}[-\.\s]?[0-9]{4}',  # US format
        r'\+?[0-9]{1,3}[-\.\s]?[0-9]{3,4}[-\.\s]?[0-9]{3,4}[-\.\s]?[0-9]{3,4}',  # International
        r'\([0-9]{3}\)\s?[0-9]{3}-[0-9]{4}'  # (123) 456-7890
    ]
    
    phones = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        phones.extend(matches)
    
    # Clean and validate
    cleaned = []
    for phone in phones:
        digits = re.sub(r'\D', '', phone)
        if 10 <= len(digits) <= 15:
            cleaned.append(phone.strip())
    
    return list(set(cleaned))

def extract_education(text):
    education = []
    lines = text.split('\n')
    
    # Look for education section
    education_section = []
    in_education = False
    
    for line in lines:
        line = line.strip()
        if re.search(r'(?i)^(education|academic|qualifications).*$', line):
            in_education = True
            continue
        elif in_education and re.search(r'(?i)^(experience|work|skills|projects|certifications).*$', line):
            break
        elif in_education:
            education_section.append(line)
    
    # Extract from education section or full text if no section found
    search_text = '\n'.join(education_section) if education_section else text
    
    # Patterns for degrees and institutions
    degree_patterns = [
        r'(?i)(bachelor|master|phd|doctorate|mba|bs|ba|ms|ma|btech|mtech).*?(?:in|of)?\s+([a-z\s]+?)(?:\d{4}|\n|$)',
        r'(?i)(university|college|institute|school)\s+of\s+([a-z\s]+)',
        r'(?i)([a-z\s]+?)\s+(university|college|institute)',
        r'(?i)(gpa|cgpa)\s*:?\s*([0-9.]+)'
    ]
    
    for pattern in degree_patterns:
        matches = re.findall(pattern, search_text)
        for match in matches:
            edu_text = ' '.join(match).strip()
            if len(edu_text) > 5 and edu_text not in education:
                education.append(edu_text)
    
    return education[:5]  # Limit to 5 entries

def extract_skills(text):
    skills = set()
    
    # Find skills section
    lines = text.split('\n')
    skills_section = []
    in_skills = False
    
    for line in lines:
        line = line.strip()
        if re.search(r'(?i)^(skills?|technical skills?|competencies).*$', line):
            in_skills = True
            continue
        elif in_skills and re.search(r'(?i)^(experience|work|education|projects|certifications).*$', line):
            break
        elif in_skills and line:
            skills_section.append(line)
    
    # Process skills section
    for line in skills_section:
        # Remove bullets and split by common separators
        clean_line = re.sub(r'^[•▪▫◦‣▸-]\s*', '', line)
        items = re.split(r'[,;|/•]|\s+and\s+|\s+&\s+', clean_line)
        
        for item in items:
            item = item.strip()
            if 2 <= len(item) <= 30 and not re.search(r'\d{4}|@|\.|www', item):
                skills.add(item)
    
    # If no skills section, look for technical terms
    if not skills:
        tech_patterns = [
            r'(?i)\b(python|java|javascript|c\+\+|c#|html|css|sql|mysql|postgresql|mongodb)\b',
            r'(?i)\b(aws|azure|docker|kubernetes|git|linux|windows|react|angular|vue)\b',
            r'(?i)\b(tensorflow|pytorch|pandas|numpy|scikit-learn|excel|tableau|powerbi)\b'
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text)
            skills.update(match.lower() for match in matches)
    
    return sorted(list(skills))

def extract_projects(text):
    projects = []
    
    # Find projects section
    lines = text.split('\n')
    project_lines = []
    in_projects = False
    
    for line in lines:
        line = line.strip()
        if re.search(r'(?i)^(projects?|personal projects?|academic projects?).*$', line):
            in_projects = True
            continue
        elif in_projects and re.search(r'(?i)^(experience|work|education|skills|certifications).*$', line):
            break
        elif in_projects and line:
            project_lines.append(line)
    
    # Extract project entries
    current_project = {}
    for line in project_lines:
        # Project titles are usually standalone lines or start with bullets
        if not line.startswith(('•', '-', '*')) and len(line.split()) <= 6:
            if current_project:
                projects.append(current_project)
            current_project = {"title": line, "description": ""}
        elif line.startswith(('•', '-', '*')) and current_project:
            desc = line[1:].strip()
            if current_project["description"]:
                current_project["description"] += " " + desc
            else:
                current_project["description"] = desc
    
    if current_project:
        projects.append(current_project)
    
    return projects[:5]  # Limit to 5 projects

def extract_certifications(text):
    certifications = []
    
    # Find certifications section
    lines = text.split('\n')
    cert_lines = []
    in_certs = False
    
    for line in lines:
        line = line.strip()
        if re.search(r'(?i)^(certifications?|certificates?|licenses?).*$', line):
            in_certs = True
            continue
        elif in_certs and re.search(r'(?i)^(experience|work|education|skills|projects).*$', line):
            break
        elif in_certs and line:
            cert_lines.append(line)
    
    # Extract certification entries
    for line in cert_lines:
        line = re.sub(r'^[•▪▫◦‣▸-]\s*', '', line).strip()
        if line and len(line) > 5:
            # Try to separate cert name and issuer/date
            if '-' in line:
                parts = line.split('-', 1)
                cert = {
                    "name": parts[0].strip(),
                    "issuer": parts[1].strip() if len(parts) > 1 else ""
                }
            else:
                cert = {"name": line, "issuer": ""}
            certifications.append(cert)
    
    return certifications[:5]  # Limit to 5 certifications


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

def calculate_work_duration(text): #name something else as it also extracts job title and company name
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
                    
                    # Look for job title in the same line or previous lines
                    # First, check if the date line itself contains the job title
                    line_before_date = line.split(match.group())[0].strip()
                    
                    # Check if this looks like a company name instead of job title
                    if line_before_date and ',' in line_before_date:
                        # If so, then true
                        potential_company = line_before_date.split(',')[0].strip()
                        if (len(potential_company.split()) <= 3 and 
                            not any(keyword.lower() in potential_company.lower() for keyword in ['engineer', 'scientist', 'analyst', 'manager', 'developer', 'specialist', 'consultant', 'architect', 'lead', 'director'])):
                            # This is likely a company, look for job title in previous lines
                            company = potential_company
                            
                            # Look for job title in previous lines
                            for j in range(max(0, i-3), i):
                                if j < len(lines):
                                    candidate_line = lines[j].strip()
                                    if (candidate_line and 
                                        len(candidate_line.split()) <= 4 and
                                        not re.search(r'\d{4}', candidate_line) and
                                        any(keyword.lower() in candidate_line.lower() for keyword in ['engineer', 'scientist', 'analyst', 'manager', 'developer', 'specialist', 'consultant', 'architect', 'lead', 'director'])):
                                        job_title = candidate_line.strip()
                                        break
                    else:
                        # Standard format - job title before date
                        if line_before_date and len(line_before_date.split()) <= 6:
                            # Remove common formatting characters and clean the title
                            clean_title = re.sub(r'^[\t\s]*', '', line_before_date)
                            clean_title = re.sub(r'[\t]+', ' ', clean_title).strip()
                            # Remove trailing tabs and extra spaces
                            clean_title = re.sub(r'[\t\s]+$', '', clean_title)
                            if clean_title and not re.search(r'\d{4}', clean_title) and len(clean_title) > 2:
                                job_title = clean_title
                    
                    # If not found, look in previous lines for job title
                    if job_title == "Unknown Title":
                        for j in range(max(0, i-3), i):
                            if j < len(lines):
                                candidate_line = lines[j].strip()
                                
                                # Skip empty lines or lines with dates
                                if not candidate_line or re.search(r'\d{4}', candidate_line):
                                    continue
                                    
                                # Check for markdown headers (##)
                                if candidate_line.startswith('##'):
                                    title_candidate = candidate_line.replace('##', '').strip()
                                    if title_candidate and len(title_candidate) < 80:
                                        job_title = title_candidate
                                        break
                                
                                # Check for lines with job keywords or typical job titles
                                job_keywords = ['engineer', 'scientist', 'analyst', 'manager', 'developer', 'specialist', 'consultant', 'architect', 'lead', 'senior', 'junior', 'associate', 'director', 'coordinator']
                                if (any(keyword.lower() in candidate_line.lower() for keyword in job_keywords) and
                                    len(candidate_line) < 80 and len(candidate_line.split()) <= 6):
                                    job_title = candidate_line.strip()
                                    break
                    
                    # Look for company in the line immediately after the date line
                    for j in range(i+1, min(len(lines), i+3)):
                        if j < len(lines):
                            candidate_line = lines[j].strip()
                            if not candidate_line:
                                continue
                                
                            # Skip lines that start with bullet points or look like job descriptions
                            if (re.search(r'^[•\*\-\u25cf➤→]', candidate_line) or
                                candidate_line.lower().startswith(('designed', 'developed', 'implemented', 'built', 'created', 'managed', 'led', 'architected', 'integrated', 'collaborated', 'enhanced', 'optimized'))):
                                continue
                            
                            # Company line typically has format: "Company Name, Location" or just "Company Name"
                            # Split by comma and take the first part as company
                            if ',' in candidate_line:
                                company_candidate = candidate_line.split(',')[0].strip()
                            else:
                                company_candidate = candidate_line.strip()
                            
                            # Validate that this looks like a company name
                            if (company_candidate and 
                                len(company_candidate.split()) <= 5 and
                                len(company_candidate) > 2 and
                                not re.search(r'\d{4}', company_candidate) and
                                not company_candidate.lower() in ['remote', 'onsite', 'hybrid']):
                                company = company_candidate
                                break
                    
                    # Fallback: Try NER for company extraction if not found
                    if company == "Unknown Company":
                        context_lines = []
                        for j in range(max(0, i-2), min(len(lines), i+3)):
                            if lines[j].strip():
                                context_lines.append(lines[j])
                        
                        context = ' '.join(context_lines)
                        doc = nlp(context)
                        orgs = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
                        
                        if orgs:
                            # Filter out common false positives
                            filtered_orgs = [org for org in orgs if not re.search(r'\d{4}', org) and 
                                           len(org.split()) <= 4 and
                                           org.lower() not in ['remote', 'present', 'current', 'environment']]
                            if filtered_orgs:
                                company = filtered_orgs[0]

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
    
    # If no formal work experience section found, check if the entire resume 
    # contains date patterns suggesting work experience scattered throughout
    if not section_found:
        # Check if there are multiple date patterns in the resume
        date_pattern = r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Sept|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}\s*(?:--|[-–—]|to)\s*(?:Present|Current|Now|Ongoing|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Sept|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})'
        date_matches = re.findall(date_pattern, text, re.IGNORECASE)
        
        # If we find multiple date ranges, assume work experience is embedded throughout
        if len(date_matches) >= 2:
            return text  # Return the full text for processing
    
    return '\n'.join(experience_text) if experience_text else None

def save_to_json(data, filename="extracted_data.json"):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Data saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving to JSON: {e}")
        return False

def parse_resume(file_path):
    # Extract text based on file type
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        text = extract_text_from_pdf(file_path)
    elif ext == '.docx':
        text = extract_text_from_docx(file_path)
    else:
        return None
    
    if not text:
        return None
    
    # Extract all information
    return {
        "name": extract_name(text),
        "emails": extract_email(text),
        "phone_numbers": extract_phone_number(text),
        "education": extract_education(text),
        "skills": extract_skills(text),
        "work_experiences": calculate_work_duration(extract_work_experience(text) or ""),
        "projects": extract_projects(text),
        "certifications": extract_certifications(text)
    }

# Test the parser with sample files
if __name__ == "__main__":
    sample_files = [
        'C:/Flexon_Resume_Parser/Parser_Build-Arnav/Test Resumes/Sample Resumes/AI Engineer.docx',
        'C:/Flexon_Resume_Parser/Parser_Build-Arnav/Test Resumes/Sample Resumes/Data Scientist_1.docx',
        'C:/Flexon_Resume_Parser/Parser_Build-Arnav/Test Resumes/Sample Resumes/AI ML Engineer.docx',
        'C:/Flexon_Resume_Parser/Parser_Build-Arnav/Test Resumes/Sample Resumes/AI_ML_Engineer_1 External.docx',
        'C:/Flexon_Resume_Parser/Parser_Build-Arnav/Test Resumes/Sample Resumes/AI_ML_Engineer_2 External.docx',
        'C:/Flexon_Resume_Parser/Parser_Build-Arnav/Test Resumes/Sample Resumes/AI_ML_Engineer_5 External.docx',
        'C:/Flexon_Resume_Parser/Parser_Build-Arnav/Test Resumes/Sample Resumes/Cloud Engineer.docx',
        'C:/Flexon_Resume_Parser/Parser_Build-Arnav/Test Resumes/Sample Resumes/Data Engineer.docx',
        'C:/Flexon_Resume_Parser/Parser_Build-Arnav/Test Resumes/Sample Resumes/Data Scientist 2.docx',
        'C:/Flexon_Resume_Parser/Parser_Build-Arnav/Test Resumes/Sample Resumes/Data_Scientist_3 External.doc',
        'C:/Flexon_Resume_Parser/Parser_Build-Arnav/Test Resumes/Sample Resumes/Data_Scientist_4 External.docx'
    ]
    
    results = {}
    
    for file_path in sample_files:
        if os.path.exists(file_path):
            #print(f"Processing: {os.path.basename(file_path)}")
            result = parse_resume(file_path)
            if result:
                results[os.path.basename(file_path)] = result
            else:
                #print(f"Failed to process: {file_path}")
                continue
    
    if results:
        save_to_json(results)
    else:
        print("No files processed successfully")
        