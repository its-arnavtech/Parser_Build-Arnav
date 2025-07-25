import argparse
import json
from main import extract_data, extract_name, extract_email, extract_phone_number, extract_urls_spacy, extract_education

def main():
    parser = argparse.ArgumentParser(description="Resume Data Extractor")
    parser.add_argument('file_path', help="Path to resume file (PDF or DOCX)")
    args = parser.parse_args()

    try:
        result = extract_data(args.file_path)

        if result:
            result_data = {
                "names": extract_name(result),
                "emails": extract_email(result),
                "phone_numbers": extract_phone_number(result),
                "urls": extract_urls_spacy(result),
                "education": extract_education(result),
            }

            print(json.dumps(result_data, indent=3))
        else:
            print("Error: No text extracted from the provided file.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
