import json

with open('extracted_data.json', 'r') as f:
    data = json.load(f)

print("=== RESUME PARSING RESULTS ANALYSIS ===\n")

for resume_file, resume_data in data['docx'].items():
    filename = resume_file.split('/')[-1]
    print(f"=== {filename} ===")
    print(f"Name: {resume_data['name']}")
    print("Work Experiences:")
    
    if resume_data['work_experiences']:
        for i, exp in enumerate(resume_data['work_experiences'], 1):
            print(f"  {i}. Job Title: {exp['job_title']}")
            print(f"     Company: {exp['company']}")
            print(f"     Duration: {exp['from']} - {exp['to']} ({exp['duration_months']} months)")
            print(f"     Current: {'Yes' if exp['is_current'] else 'No'}")
            print()
    else:
        print("  No work experiences found")
    
    print(f"Total Experience: {resume_data['total_experience_years']} years")
    print(f"Education: {len(resume_data['education'])} entries")
    print(f"Skills: {len(resume_data['skills'])} entries")
    print(f"Emails: {resume_data['emails']}")
    print(f"Phone: {resume_data['phone_numbers']}")
    print("-" * 80)
    print()
