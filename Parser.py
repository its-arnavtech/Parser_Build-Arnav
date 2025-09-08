import os
import spacy # pyright: ignore[reportMissingImports]
import re
import json
from pdfminer.high_level import extract_text # pyright: ignore[reportMissingImports]
from docx import Document # pyright: ignore[reportMissingImports]
from datetime import datetime
from dateutil.relativedelta import relativedelta
import dateparser # pyright: ignore[reportMissingModuleSource]

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Common job title keywords for better detection
JOB_KEYWORDS = [
    'engineer', 'scientist', 'analyst', 'manager', 'developer', 'specialist', 'consultant',
    'architect', 'lead', 'senior', 'junior', 'associate', 'director', 'coordinator',
    'administrator', 'designer', 'researcher', 'intern', 'assistant', 'supervisor'
]

# Common company/organization indicators
COMPANY_INDICATORS = ['inc', 'corp', 'ltd', 'llc', 'co', 'company', 'corporation', 'limited', 'technologies', 'tech', 'solutions', 'systems', 'group', 'international']

# Date patterns for work experience
DATE_PATTERNS = [
    r'(?P<from>\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Sept|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\s*(?:--|[-–—]|to)\s*(?P<to>(?:Present|Current|Now|Ongoing|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Sept|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}))',
    r'(?P<from>\b\d{4})\s*(?:--|[-–—]|to)\s*(?P<to>(?:Present|Current|Now|Ongoing|\b\d{4}))',
    r'(?P<from>\b\d{1,2}/\d{4})\s*(?:--|[-–—]|to)\s*(?P<to>(?:Present|Current|Now|Ongoing|\b\d{1,2}/\d{4}))'
]

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
    lines = text.strip().split('\n')[:30]  #check first 30 lines
    
    # Look for **Name** pattern first
    for line in lines[:10]:
        line = line.strip()
        if re.match(r'^\*\*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\*\*$', line):
            return re.match(r'^\*\*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\*\*$', line).group(1)
    
    # Use NLP for person detection
    doc = nlp(' '.join(lines))
    for ent in doc.ents:
        if ent.label_ == "PERSON" and len(ent.text.split()) <= 3:
            return ent.text.strip()
    
    # Regex fallback
    for line in lines:
        line = line.strip()
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
    
    cleaned = []
    for phone in phones:
        digits = re.sub(r'\D', '', phone)
        if 10 <= len(digits) <= 15:
            cleaned.append(phone.strip())
    
    return list(set(cleaned))

def extract_education(text):
    education = []
    lines = text.split('\n')
    
    # Find education section
    education_section = []
    in_education = False
    
    for line in lines:
        line = line.strip()
        if re.search(r'(?i)^(education|academic|qualifications).*$', line):
            in_education = True
            continue
        elif in_education and re.search(r'(?i)^(experience|work|skills|projects|certifications|relevant\s+course).*$', line):
            break
        elif in_education:
            education_section.append(line)
    
    # Use education data or search full resume
    search_text = '\n'.join(education_section) if education_section else text
    
    # Improved degree patterns - look for actual degrees in the resume
    degree_patterns = [
        r'(?i)(M\.?S\.?\s+\([^)]+\))',  # M.S (Electrical and Computer Engineering)
        r'(?i)(M\.?E\.?\s+\([^)]+\))',  # M.E (Embedded System Technologies) 
        r'(?i)(B\.?E\.?\s+\([^)]+\))',  # B.E (Electronics and Communication Engineering)
        r'(?i)(bachelor|master|phd|doctorate|mba|bs|ba|ms|ma|btech|mtech)\s+.*?(?:in|of)?\s+([a-zA-Z\s&]+?)(?:\.|,|\n|$)',
        r'(?i)(university|college|institute|school)\s+.*?(\d{4})',
    ]
    
    for pattern in degree_patterns:
        matches = re.findall(pattern, search_text)
        for match in matches:
            if isinstance(match, tuple):
                edu_text = ' '.join(filter(None, match)).strip()
            else:
                edu_text = match.strip()
            
            if len(edu_text) > 5 and edu_text not in education:
                education.append(edu_text)
    
    return education[:5]  # max 5 entries

def extract_skills(text):
    skills = set()
    
    # Find technical skills section
    lines = text.split('\n')
    skills_section = []
    in_skills = False
    
    for line in lines:
        line = line.strip()
        if re.search(r'(?i)^(technical\s+skills?|skills?|competencies).*$', line):
            in_skills = True
            continue
        elif in_skills and re.search(r'(?i)^(education|experience|work|projects|certifications).*$', line):
            break
        elif in_skills and line:
            skills_section.append(line)
    
    # Parse skills section with better categorization
    for line in skills_section:
        # Remove bullet points and clean line
        clean_line = re.sub(r'^[•▪▫◦‣▸-]\s*', '', line)
        
        # Handle categorized skills format (e.g., "Programming Languages Python, SQL, R...")
        category_match = re.match(r'^([^:]+):\s*(.+)', clean_line)
        if category_match:
            skills_text = category_match.group(2)
        else:
            skills_text = clean_line
        
        # Split by various delimiters
        items = re.split(r'[,;|/•]|\s+and\s+|\s+&\s+', skills_text)
        
        for item in items:
            item = item.strip()
            # Better filtering for valid skills
            if (2 <= len(item) <= 35 and 
                not re.search(r'\d{4}|@|\.|www|http', item) and
                not item.lower() in ['and', 'or', 'with', 'using', 'including']):
                skills.add(item)
    
    # If no skills section found, look for technical skills in full text
    if not skills:
        tech_patterns = [
            r'(?i)\b(python|java|javascript|c\+\+|c#|html|css|sql|mysql|postgresql|mongodb|r|sas|matlab)\b',
            r'(?i)\b(aws|azure|docker|kubernetes|git|linux|windows|react|angular|vue|tableau|power\s*bi)\b',
            r'(?i)\b(tensorflow|pytorch|pandas|numpy|scikit-learn|keras|spark|hadoop|snowflake)\b'
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text)
            skills.update(match.strip() for match in matches)
    
    return sorted(list(skills))

def extract_projects(text):
    """Extract projects with improved detection and parsing"""
    projects = []
    lines = text.split('\n')
    
    # Find projects section
    project_section = []
    in_projects = False
    
    for line in lines:
        line_clean = line.strip()
        if re.match(r'(?i)^(projects?|personal\s*projects?|academic\s*projects?|key\s*projects?)[:*\-\s]*$', line_clean):
            in_projects = True
            continue
        elif in_projects and re.match(r'(?i)^(experience|work|employment|education|skills|certifications?|awards?|achievements?)[:*\-\s]*$', line_clean):
            break
        elif in_projects and line_clean:
            project_section.append(line.rstrip())
    
    if not project_section:
        return projects
    
    # Parse projects from section
    current_project = None
    bullet_patterns = [r'^[\u2022\*\-\u25cf\u27a4]', r'^\d+\.', r'^[a-zA-Z]\)']
    
    for line in project_section:
        line_clean = line.strip()
        if not line_clean:
            continue
        
        # Check if this is a bullet point (description)
        is_bullet = any(re.match(pattern, line_clean) for pattern in bullet_patterns)
        
        if not is_bullet and len(line_clean.split()) <= 10 and not line_clean.startswith(' '):
            # This looks like a project title
            if current_project and current_project.get('title'):
                projects.append(current_project)
            
            # Clean the title
            title = re.sub(r'^[\u2022\*\-\u25cf\u27a4\d\.\s]+', '', line_clean).strip()
            title = re.sub(r'[:]+$', '', title)  # Remove trailing colons
            
            current_project = {
                "title": title,
                "description": "",
                "technologies": []
            }
        
        elif current_project and (is_bullet or line_clean.startswith('  ')):
            # This is a description line
            desc_text = re.sub(r'^[\u2022\*\-\u25cf\u27a4\d\.\s]+', '', line_clean).strip()
            
            # Check for technologies in parentheses or after keywords
            tech_match = re.search(r'(?:technologies?|tech stack|tools?)[:]*\s*([^.]+)', desc_text, re.IGNORECASE)
            if tech_match:
                tech_text = tech_match.group(1)
                techs = re.split(r'[,;|&]|\sand\s', tech_text)
                current_project["technologies"].extend([t.strip() for t in techs if t.strip()])
            
            if current_project["description"]:
                current_project["description"] += " " + desc_text
            else:
                current_project["description"] = desc_text
    
    # Add the last project
    if current_project and current_project.get('title'):
        projects.append(current_project)
    
    # Clean up descriptions and validate projects
    cleaned_projects = []
    for project in projects:
        if project["title"] and len(project["title"]) > 2:
            # Limit description length
            if len(project["description"]) > 500:
                project["description"] = project["description"][:500] + "..."
            
            # Remove duplicates from technologies
            project["technologies"] = list(set(project["technologies"]))
            
            cleaned_projects.append(project)
    
    return cleaned_projects[:8]  # Max 8 projects

def extract_certifications(text):
    """Extract certifications with improved parsing for various formats"""
    certifications = []
    lines = text.split('\n')
    
    # Find certifications section
    cert_section = []
    in_certs = False
    
    for line in lines:
        line_clean = line.strip()
        if re.match(r'(?i)^(certifications?|certificates?|licenses?|professional\s*certifications?|trainings?)[:*\-\s]*$', line_clean):
            in_certs = True
            continue
        elif in_certs and re.match(r'(?i)^(experience|work|employment|education|skills|projects?|awards?|achievements?)[:*\-\s]*$', line_clean):
            break
        elif in_certs and line_clean:
            cert_section.append(line_clean)
    
    if not cert_section:
        # Try to find certifications in the full text using common patterns
        cert_patterns = [
            r'(?i)(aws|amazon|microsoft|google|oracle|cisco|comptia|pmp|cissp|cisa|cism)\s+(certified|certification)',
            r'(?i)(certified|certification)\s+.{5,50}',
            r'(?i).{5,50}\s+(certified|certification)',
        ]
        
        for pattern in cert_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                cert_text = ' '.join(match) if isinstance(match, tuple) else match
                if len(cert_text) > 10 and len(cert_text) < 100:
                    certifications.append({
                        "name": cert_text.strip(),
                        "issuer": "",
                        "date": ""
                    })
        
        return certifications[:10]
    
    # Parse certifications from section
    for line in cert_section:
        line = re.sub(r'^[\u2022\*\-\u25cf\u27a4\d\.\s]+', '', line).strip()
        
        if len(line) < 5 or len(line) > 150:
            continue
        
        cert_entry = {
            "name": "",
            "issuer": "",
            "date": ""
        }
        
        # Try different formats
        # Format: "Cert Name - Issuer (Date)" or "Cert Name - Issuer, Date"
        if ' - ' in line:
            parts = line.split(' - ', 1)
            cert_entry["name"] = parts[0].strip()
            
            remaining = parts[1].strip()
            # Look for date in parentheses or after comma
            date_match = re.search(r'\(?([A-Za-z]+\s*\d{4}|\d{4}|\d{1,2}/\d{4}|\d{1,2}-\d{4})\)?', remaining)
            if date_match:
                cert_entry["date"] = date_match.group(1)
                cert_entry["issuer"] = re.sub(r'\s*[,\(].*', '', remaining).strip()
            else:
                cert_entry["issuer"] = remaining
        
        # Format: "Cert Name (Issuer)" or "Cert Name | Issuer"
        elif '(' in line and ')' in line:
            match = re.match(r'^(.+?)\s*\((.+?)\)\s*(.*)', line)
            if match:
                cert_entry["name"] = match.group(1).strip()
                cert_entry["issuer"] = match.group(2).strip()
                if match.group(3):
                    cert_entry["date"] = match.group(3).strip()
        
        # Format: "Cert Name | Issuer" or "Cert Name / Issuer"
        elif '|' in line or '/' in line:
            separator = '|' if '|' in line else '/'
            parts = line.split(separator, 1)
            cert_entry["name"] = parts[0].strip()
            if len(parts) > 1:
                cert_entry["issuer"] = parts[1].strip()
        
        # Format: "Cert Name, Issuer" 
        elif ',' in line:
            parts = line.split(',', 1)
            cert_entry["name"] = parts[0].strip()
            remaining = parts[1].strip()
            
            # Check if remaining part looks like a date
            if re.match(r'^\d{4}|[A-Za-z]+\s*\d{4}', remaining):
                cert_entry["date"] = remaining
            else:
                cert_entry["issuer"] = remaining
        
        # Simple format: just the certification name
        else:
            cert_entry["name"] = line
        
        # Validate and add certification
        if cert_entry["name"] and len(cert_entry["name"]) > 3:
            # Clean up empty fields
            for key in cert_entry:
                if cert_entry[key] == "":
                    cert_entry[key] = None
            
            certifications.append(cert_entry)
    
    return certifications[:12]  # Max 12 certifications

def parse_date(date_str):
    """Helper function to parse various date formats"""
    try:
        normalized = date_str.replace('Sept', 'Sep').strip()
        return dateparser.parse(normalized)
    except:
        return None

def calculate_duration(start_date, end_date):
    """Calculate duration in months between two dates"""
    if not start_date or not end_date:
        return 0
    duration = relativedelta(end_date, start_date)
    return max(0, duration.years * 12 + duration.months + (1 if duration.days > 15 else 0))

def extract_work_experience_section(text):
    """Extract the professional experience section from resume text"""
    section_headers = [
        r"(?i)work[\s\*\[\]]*experience",
        r"(?i)professional[\s\*\[\]]*experience", 
        r"(?i)employment[\s\*\[\]]*history",
        r"(?i)career[\s\*\[\]]*history",
        r"(?i)experience[\s\*\[\]]*:",
        r"(?i)\*\*.*professional.*experience.*\*\*",
        r"(?i)professional[\s\*\[\]]*experiences"
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
        r"(?i)\*\*.*education.*\*\*",
        r"(?i)other[\s\*\[\]]*experiences"
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
    
    # If no formal work experience section found, return full text
    if not section_found:
        return text
    
    return '\n'.join(experience_text) if experience_text else text

def extract_work_experiences(text):
    """Enhanced work experience extraction focusing on the actual resume format"""
    if not text:
        return []
    
    jobs = []
    lines = text.split('\n')
    
    # Look for professional experience entries with company, location, URL, and dates
    experience_patterns = [
        r'([A-Za-z\s&,]+(?:Inc|LLC|Corp|Ltd|Technologies|Health)),\s*([A-Z]{2,})\.\s*(https?://[^\s]+)\s+([A-Za-z]{3}\s+\d{4})\s*[-–—]\s*([A-Za-z]{3}\s+\d{4}|Present)',
        r'([A-Za-z\s&,]+),\s*([A-Z]{2,})\.\s*(https?://[^\s]+)\s+([A-Za-z]{3}\s+\d{4})\s*[-–—]\s*([A-Za-z]{3}\s+\d{4}|Present)'
    ]
    
    for i, line in enumerate(lines):
        line_clean = line.strip()
        
        for pattern in experience_patterns:
            match = re.search(pattern, line_clean)
            if match:
                company = match.group(1).strip()
                location = match.group(2).strip()
                url = match.group(3)
                start_date_str = match.group(4)
                end_date_str = match.group(5)
                
                # Parse dates
                start_date = parse_date(start_date_str)
                if end_date_str.lower() in ['present', 'current']:
                    end_date = datetime.now()
                    is_current = True
                else:
                    end_date = parse_date(end_date_str)
                    is_current = False
                
                if start_date:
                    duration_months = calculate_duration(start_date, end_date)
                    
                    # Look for job title in next lines
                    job_title = "Data Scientist"  # Default
                    for j in range(i + 1, min(len(lines), i + 4)):
                        next_line = lines[j].strip()
                        # Skip empty lines and check for job title patterns
                        if (next_line and not next_line.startswith(('•', '-', '*', 'Responsibilities')) and
                            len(next_line.split()) <= 5 and not next_line.startswith('http')):
                            job_title = re.sub(r'^\*\*|\*\*$', '', next_line).strip()
                            break
                    
                    # Extract job description
                    description_lines = []
                    desc_started = False
                    for j in range(i + 1, min(len(lines), i + 30)):
                        desc_line = lines[j].strip()
                        if not desc_line:
                            continue
                        
                        if desc_line.startswith(('Responsibilities:', '**Responsibilities:**')):
                            desc_started = True
                            continue
                        
                        if desc_started and desc_line.startswith(('•', '-')):
                            clean_desc = re.sub(r'^[•\-]\s*', '', desc_line).strip()
                            if len(clean_desc) > 20:
                                description_lines.append(clean_desc)
                                if len(description_lines) >= 5:
                                    break
                        elif (desc_line.startswith(('Environment:', '**Environment:**')) or
                              re.match(r'[A-Za-z\s&,]+(?:Inc|LLC|Corp|Ltd)', desc_line)):
                            break
                    
                    work_description = ' '.join(description_lines) if description_lines else "No description available"
                    
                    job_entry = {
                        "job_title": job_title,
                        "company": company,
                        "duration_months": duration_months,
                        "from": start_date.strftime('%b %Y'),
                        "to": "Present" if is_current else end_date.strftime('%b %Y'),
                        "is_current": is_current,
                        "work_description": work_description
                    }
                    
                    jobs.append(job_entry)
    
    # Sort by start date (most recent first)
    jobs.sort(key=lambda x: datetime.strptime(x['from'], '%b %Y'), reverse=True)
    return jobs

def total_experience(jobs):
    """Calculate total experience in years"""
    if not jobs:
        return 0
    total_months = sum(job["duration_months"] for job in jobs)
    return round(total_months / 12, 2)

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
    """Main function to parse resume and extract all information"""
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
    
    # Extract work experience section or use full text if no section found
    work_experience_text = extract_work_experience_section(text)
    
    # Extract all information
    return {
        "name": extract_name(text),
        "emails": extract_email(text),
        "phone_numbers": extract_phone_number(text),
        "education": extract_education(text),
        "skills": extract_skills(text),
        "work_experiences": extract_work_experiences(work_experience_text),
        "total_experience_years": total_experience(extract_work_experiences(work_experience_text)),
        "projects": extract_projects(text),
        "certifications": extract_certifications(text)
    }

# Test the parser with sample files
if __name__ == "__main__":
    sample_files = [
        #'C:/Flexon_Resume_Parser/Parser_Build-Arnav/Test Resumes/Sample Resumes/AI Engineer.docx',
        #'C:/Flexon_Resume_Parser/Parser_Build-Arnav/Test Resumes/Sample Resumes/Data Scientist_1.docx',
        #'C:/Flexon_Resume_Parser/Parser_Build-Arnav/Test Resumes/Sample Resumes/AI ML Engineer.docx',
        #'C:/Flexon_Resume_Parser/Parser_Build-Arnav/Test Resumes/Sample Resumes/AI_ML_Engineer_1 External.docx',
        #'C:/Flexon_Resume_Parser/Parser_Build-Arnav/Test Resumes/Sample Resumes/AI_ML_Engineer_2 External.docx',
        #'C:/Flexon_Resume_Parser/Parser_Build-Arnav/Test Resumes/Sample Resumes/AI_ML_Engineer_5 External.docx',
        #'C:/Flexon_Resume_Parser/Parser_Build-Arnav/Test Resumes/Sample Resumes/Cloud Engineer.docx',
        #'C:/Flexon_Resume_Parser/Parser_Build-Arnav/Test Resumes/Sample Resumes/Data Engineer.docx',
        #'C:/Flexon_Resume_Parser/Parser_Build-Arnav/Test Resumes/Sample Resumes/Data Scientist 2.docx',
        #'C:/Flexon_Resume_Parser/Parser_Build-Arnav/Test Resumes/Sample Resumes/Data_Scientist_3 External.doc',
        'C:/Flexon_Resume_Parser/Parser_Build-Arnav/Test Resumes/Sample Resumes/Data_Scientist_4 External.docx'
        #'C:/Flexon_Resume_Parser/Parser_Build-Arnav/Test Resumes/Resume_ArnavK.pdf'
    ]
    
    results = {}
    
    for file_path in sample_files:
        if os.path.exists(file_path):
            result = parse_resume(file_path)
            if result:
                results[os.path.basename(file_path)] = result
            else:
                print(f"Failed to process: {file_path}")
                continue
    
    if results:
        save_to_json(results)
    else:
        print("No files processed successfully")
        