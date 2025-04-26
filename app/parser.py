import re
import spacy
import fitz  # PyMuPDF
from pathlib import Path
from time import time
from pprint import pprint

from app.models import extract_skills_with_model  # Now model-driven skill extraction

nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_email(text):
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    return email_match.group(0) if email_match else None

def extract_phone(text):
    phone_match = re.search(r'\+?\d[\d\-\s]{8,}\d', text)
    return phone_match.group(0) if phone_match else None

def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None

def parse_resume(file_path):
    raw_text = extract_text_from_pdf(file_path)

    parsed_data = {
        "name": extract_name(raw_text),
        "email": extract_email(raw_text),
        "phone": extract_phone(raw_text),
        "skills": extract_skills_with_model(raw_text),
        "raw_text": raw_text[:1000]  # Optional preview
    }
    return parsed_data

# Local test
if __name__ == "__main__":
    test_file = Path("data/resumes/sample_resume.pdf")
    pprint(parse_resume(test_file), sort_dicts=False)
    start = time()
    result = parse_resume(test_file)
    end = time()
    print(f"\nTime taken: {round(end - start, 2)} seconds")

