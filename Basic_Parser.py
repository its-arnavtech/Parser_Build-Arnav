import os
import spacy
import re
import json
from pdfminer.high_level import extract_text
from docx import Document
from datetime import datetime
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse

#using logger as suggested, setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('resume_parser.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    nlp = spacy.load("en_core_web_md")
    logger.info("loaded spaCy model")
except OSError:
    logger.error("spaCy model not found")
    raise

def extract_text_from_pdf(pdf_path):
    try:
        text = extract_text(pdf_path)
        if not text or len(text.strip()) < 10:
            logger.warning(f"PDF not found: {pdf_path}")
            return None
        return text
    except Exception as e:
        logger.error(f"Extraction error {pdf_path}: {str(e)}")
        return None

def extract_text_from_docx(docx_path):
    try:
        doc = Document(docx_path)
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        if not text or len(text.strip()) < 10:
            logger.warning(f"Document Error: {docx_path}")
            return None
        return text
    except Exception as e:
        logger.error(f"Extraction error {docx_path}: {str(e)}")
        return None

def extract_name(text):
    lines = text.strip().split('\n')[:20]
    
    #spacy first
    try:
        doc = nlp(' '.join(lines))
        for ent in doc.ents:
            if ent.label_ == "PERSON" and 2 <= len(ent.text.split()) <= 4:
                name_parts = ent.text.strip().split()
                if len(name_parts) >= 2:
                    first_name = name_parts[0]
                    last_name = name_parts[-1]
                    return first_name, last_name
    except Exception as e:
        logger.debug(f"SpaCy NER failed: {str(e)}")
    
    #regex for capitalized names
    for line in lines:
        line = line.strip()
        match = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)*[A-Z][a-z]+)(?:\s|$)', line)
        if match and not re.search(r'\d|@|\.com|phone|email|address', match.group(1).lower()):
            name_parts = match.group(1).split()
            if len(name_parts) >= 2:
                filtered_parts = [part for part in name_parts if len(part) > 1 and part not in ['Jr', 'Sr', 'III', 'II']]
                if len(filtered_parts) >= 2:
                    first_name = filtered_parts[0]
                    last_name = filtered_parts[-1]
                    return first_name, last_name
    
    return None, None

def extract_email(text):
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(pattern, text, re.IGNORECASE)
    #check validity
    valid_emails = []
    for email in set(emails):
        if not re.search(r'(example|test|sample|dummy)\.com', email.lower()):
            valid_emails.append(email.lower())
    return valid_emails

def extract_phone_number(text):
    patterns = [
        r'\+?1?[-\.\s]?\(?[0-9]{3}\)?[-\.\s]?[0-9]{3}[-\.\s]?[0-9]{4}',  # US format
        r'\+?[0-9]{1,3}[-\.\s]?[0-9]{3,4}[-\.\s]?[0-9]{3,4}[-\.\s]?[0-9]{3,4}',  # International
        r'\([0-9]{3}\)\s?[0-9]{3}-[0-9]{4}',  # (123) 456-7890
        r'[0-9]{3}[-\.\s][0-9]{3}[-\.\s][0-9]{4}'  # 123-456-7890
    ]
    
    phones = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        phones.extend(matches)

    cleaned = []
    for phone in phones:
        #remove non digits
        digits = re.sub(r'\D', '', phone)
        #check validity
        if 10 <= len(digits) <= 15:
            #keep original format
            cleaned_phone = re.sub(r'\s+', ' ', phone.strip())
            if cleaned_phone not in cleaned:
                cleaned.append(cleaned_phone)
    
    return cleaned

def extract_linkedin(text):
    patterns = [
        r'linkedin\.com/in/[\w\-]+/?',
        r'www\.linkedin\.com/in/[\w\-]+/?',
        r'https?://(?:www\.)?linkedin\.com/in/[\w\-]+/?'
    ]
    
    linkedin_urls = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if not match.startswith('http'):
                match = 'https://' + match
            linkedin_urls.append(match)
    
    return list(set(linkedin_urls))

def clean_content(text, name_parts=None, emails=None, phones=None, linkedin_urls=None):
    if not text:
        return ""
    
    #remove extracted text
    content = text
    
    #remove names
    if name_parts:
        first_name, last_name = name_parts
        if first_name and last_name:
            name_patterns = [
                rf'\b{re.escape(first_name)}\s+{re.escape(last_name)}\b',
                rf'\b{re.escape(last_name)},\s+{re.escape(first_name)}\b',
                rf'\b{re.escape(first_name)}\b',
                rf'\b{re.escape(last_name)}\b'
            ]
            for pattern in name_patterns:
                content = re.sub(pattern, '', content, flags=re.IGNORECASE)
    
    #remove emails
    if emails:
        for email in emails:
            content = content.replace(email, '')
    
    #remove phones
    if phones:
        for phone in phones:
            content = content.replace(phone, '')
    
    #remove linkedin urls
    if linkedin_urls:
        for url in linkedin_urls:
            content = content.replace(url, '')
    
    #remove common contact headers
    contact_headers = [
        r'contact\s*information?', r'personal\s*information?', r'contact\s*details?',
        r'phone:', r'email:', r'linkedin:', r'address:', r'location:'
    ]
    for header in contact_headers:
        content = re.sub(header, '', content, flags=re.IGNORECASE)
    
    content = re.sub(r'[•▪▫◦‣⁃]\s*', '', content)  #remove bullet points
    content = re.sub(r'[-*+]\s+', '', content)  #remove more bullet point types
    content = re.sub(r'\s*[,;]\s*', ', ', content)  #normalize commas
    content = re.sub(r'\n\s*\n', '\n', content)  #clean linebreaks
    content = re.sub(r'\s+', ' ', content)  #clean whitespace
    content = re.sub(r'[^\w\s,.\-()&/]', ' ', content)  #remove special charecters
    
    #remove short lines
    lines = content.split('\n')
    cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 3]
    
    return '\n'.join(cleaned_lines).strip()

def parse_resume(file_path):
    try:
        logger.info(f"processing: {file_path}")
        
        ext = Path(file_path).suffix.lower()
        if ext == '.pdf':
            text = extract_text_from_pdf(file_path)
        elif ext in ['.docx', '.doc']:
            text = extract_text_from_docx(file_path)
        else:
            logger.warning(f"Filetype not supported: {file_path}")
            return None
        
        if not text:
            logger.warning(f"No data taken from: {file_path}")
            return None
        
        #extract main info
        first_name, last_name = extract_name(text)
        emails = extract_email(text)
        phone_numbers = extract_phone_number(text)
        linkedin_urls = extract_linkedin(text)
        
        #remaining content
        cleaned_content = clean_content(
            text, 
            (first_name, last_name), 
            emails, 
            phone_numbers, 
            linkedin_urls
        )
        
        result = {
            "first_name": first_name,
            "last_name": last_name,
            "email": emails[0] if emails else None,
            "phone_number": phone_numbers[0] if phone_numbers else None,
            "linkedin": linkedin_urls[0] if linkedin_urls else None,
            "cleaned_content": cleaned_content
        }
        
        logger.info(f"processed: {file_path}")
        return result
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def get_resume_files(folder_path):
    supported_extensions = ['.pdf', '.docx', '.doc']
    resume_files = []
    
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder does not exist: {folder_path}")
        return []
    
    for ext in supported_extensions:
        files = list(folder.glob(f'*{ext}'))
        resume_files.extend(files)
    
    return resume_files

def save_to_json(data, filename="extracted_resume_data.json"):
    try:
        #output extra details - metadata
        output_data = {
            "metadata": {
                "extraction_date": datetime.now().isoformat(),
                "total_resumes": len(data),
                "successful_extractions": len([v for v in data.values() if v is not None])
            },
            "resumes": data
        }
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Data saved to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving to JSON: {e}")
        return False

def process_resumes_parallel(file_paths, max_workers=4):
    #resumes processed in parallel
    results = {}
    failed_files = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        #submit all tasks
        future_to_file = {executor.submit(parse_resume, file_path): file_path 
                         for file_path in file_paths}
        
        #collect completed results
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                if result:
                    results[Path(file_path).name] = result
                else:
                    failed_files.append(str(file_path))
            except Exception as e:
                logger.error(f"parallel processing error for {file_path}: {str(e)}")
                failed_files.append(str(file_path))
    
    return results, failed_files

def main():
    #process all resu,es
    parser = argparse.ArgumentParser(description='Extract information from resume files')
    parser.add_argument('folder_path', nargs='?', default=None, help='Path to folder containing resume files')
    parser.add_argument('--output', '-o', default='extracted_resume_data.json', 
                       help='Output JSON file name (default: extracted_resume_data.json)')
    parser.add_argument('--workers', '-w', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    
    args = parser.parse_args()
    
    if args.folder_path is None:
        args.folder_path = "C:/Flexon_Resume_Parser/Parser_Build-Arnav/Mani"
    
    resume_files = get_resume_files(args.folder_path)
    
    if not resume_files:
        logger.error("No resumes found")
        return
    
    logger.info(f"Starting to process {len(resume_files)} resume files...")
    
    #processing
    start_time = datetime.now()
    results, failed_files = process_resumes_parallel(resume_files, max_workers=args.workers)
    end_time = datetime.now()
    
    #logged results
    logger.info(f"Processing completed in {end_time - start_time}")
    logger.info(f"Successfully processed: {len(results)} files")
    logger.info(f"Failed to process: {len(failed_files)} files")
    
    if failed_files:
        logger.info("Failed files:")
        for file_path in failed_files:
            logger.info(f"  - {file_path}")
    
    if results:
        if save_to_json(results, args.output):
            logger.info(f"All data successfully saved to {args.output}")
        else:
            logger.error("Failed to save data to JSON")
    else:
        logger.error("No files processed successfully")

if __name__ == "__main__":
    main()