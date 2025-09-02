import os
import spacy
import re
from docx import Document 
from datetime import datetime
from dateutil.relativedelta import relativedelta
import dateparser

nlp = spacy.load("en_core_web_md")

def extract_text_from_docx(docx_path):
    try:
        doc = Document(docx_path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        return None

def debug_extract_work_experience(text, filename):
    print(f"\n=== DEBUG: {filename} ===")
    
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
    
    print(f"Total lines in resume: {len(lines)}")
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        if not recording:
            for header_pattern in section_headers:
                if re.search(header_pattern, line_stripped):
                    print(f"Found work experience section at line {i}: '{line_stripped}'")
                    recording = True
                    section_found = True
                    break
        
        if recording:
            section_ended = False
            for next_pattern in next_section_keywords:
                if re.search(next_pattern, line_stripped):
                    print(f"Work experience section ended at line {i}: '{line_stripped}'")
                    recording = False
                    section_ended = True
                    break
            
            if recording and not section_ended:
                if line.strip():
                    experience_text.append(line)
    
    print(f"Work experience section found: {section_found}")
    print(f"Lines extracted: {len(experience_text)}")
    
    if experience_text:
        print("First 10 lines of work experience section:")
        for i, line in enumerate(experience_text[:10]):
            print(f"  {i+1}: '{line.strip()}'")
    
    return '\n'.join(experience_text) if experience_text else None

def debug_calculate_work_duration(text, filename):
    if not text:
        print(f"No work experience text for {filename}")
        return []
    
    print(f"\nDEBUG: Processing work duration for {filename}")
    
    date_patterns = [
        r'(?P<from>\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Sept|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\s*(?:--|[-–—]|to)\s*(?P<to>(?:Present|Current|Now|Ongoing|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Sept|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}))',
        r'(?P<from>\b\d{4})\s*(?:--|[-–—]|to)\s*(?P<to>(?:Present|Current|Now|Ongoing|\b\d{4}))',
        r'(?P<from>\b\d{1,2}/\d{4})\s*(?:--|[-–—]|to)\s*(?P<to>(?:Present|Current|Now|Ongoing|\b\d{1,2}/\d{4}))',
    ]

    jobs = []
    lines = text.split('\n')
    
    print(f"Looking for date patterns in {len(lines)} lines")

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        for pattern_idx, pattern in enumerate(date_patterns):
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                print(f"Found date match on line {i} (pattern {pattern_idx+1}): '{line}'")
                print(f"  From: {match.group('from')}, To: {match.group('to')}")
                
                # Look for job title and company
                job_title = "Unknown Title"
                company = "Unknown Company"
                
                # Check the same line before the date
                line_before_date = line.split(match.group())[0].strip()
                if line_before_date:
                    print(f"  Potential job title from same line: '{line_before_date}'")
                    if len(line_before_date.split()) <= 6 and len(line_before_date) > 2:
                        clean_title = re.sub(r'[\t\s]+', ' ', line_before_date).strip()
                        if not re.search(r'\d{4}', clean_title):
                            job_title = clean_title
                            print(f"  -> Job title set to: '{job_title}'")
                
                # Look for company in next lines
                for j in range(i+1, min(len(lines), i+3)):
                    if j < len(lines):
                        candidate_line = lines[j].strip()
                        if candidate_line:
                            print(f"  Checking line {j} for company: '{candidate_line}'")
                            
                            if not (re.search(r'^[•\*\-\u25cf➤→]', candidate_line) or
                                   candidate_line.lower().startswith(('designed', 'developed', 'implemented', 'built', 'created', 'managed', 'led', 'architected', 'integrated', 'collaborated', 'enhanced', 'optimized'))):
                                
                                if ',' in candidate_line:
                                    company_candidate = candidate_line.split(',')[0].strip()
                                else:
                                    company_candidate = candidate_line.strip()
                                
                                if (company_candidate and 
                                    len(company_candidate.split()) <= 5 and
                                    len(company_candidate) > 2 and
                                    not re.search(r'\d{4}', company_candidate) and
                                    not company_candidate.lower() in ['remote', 'onsite', 'hybrid']):
                                    company = company_candidate
                                    print(f"  -> Company set to: '{company}'")
                                    break
                
                try:
                    from_str = match.group('from')
                    to_str = match.group('to')
                    
                    from_str_normalized = from_str.replace('Sept', 'Sep')
                    start = dateparser.parse(from_str_normalized)
                    
                    present_keywords = ['present', 'current', 'now', 'ongoing']
                    if any(keyword in to_str.lower() for keyword in present_keywords):
                        today = datetime.now()
                        end = datetime(today.year, today.month, 1)
                    else:
                        to_str_normalized = to_str.replace('Sept', 'Sep')
                        end = dateparser.parse(to_str_normalized)
                    
                    if start and end:
                        duration = relativedelta(end, start)
                        months = duration.years * 12 + duration.months
                        
                        job_entry = {
                            "job_title": job_title,
                            "company": company,
                            "duration_months": round(months, 1),
                            "from": start.strftime('%b %Y'),
                            "to": end.strftime('%b %Y') if not any(keyword in to_str.lower() for keyword in present_keywords) else "Present",
                            "is_current": any(keyword in to_str.lower() for keyword in present_keywords)
                        }
                        
                        jobs.append(job_entry)
                        print(f"  -> Added job entry: {job_title} at {company}")
                        
                except Exception as e:
                    print(f"  -> Error parsing dates: {e}")
                    
                break  # Found a pattern, stop checking other patterns for this line
    
    print(f"Total jobs found: {len(jobs)}")
    return jobs

# Test the problematic resumes
problematic_files = [
    'C:/Flexon_Resume_Parser/Parser_Build-Arnav/Test Resumes/Sample Resumes/Cloud Engineer.docx',
    'C:/Flexon_Resume_Parser/Parser_Build-Arnav/Test Resumes/Sample Resumes/Data Scientist AI_ML Engineer.docx'
]

for file_path in problematic_files:
    filename = file_path.split('/')[-1]
    resume_text = extract_text_from_docx(file_path)
    
    if resume_text:
        print(f"\n{'='*60}")
        print(f"DEBUGGING: {filename}")
        print(f"{'='*60}")
        
        # Show first 20 lines of resume
        lines = resume_text.split('\n')
        print(f"First 20 lines of resume:")
        for i, line in enumerate(lines[:20]):
            print(f"{i+1:2d}: '{line.strip()}'")
        
        experience_section_text = debug_extract_work_experience(resume_text, filename)
        
        if experience_section_text:
            work_experiences = debug_calculate_work_duration(experience_section_text, filename)
        else:
            print("No work experience section found - checking entire resume for date patterns")
            work_experiences = debug_calculate_work_duration(resume_text, filename)
