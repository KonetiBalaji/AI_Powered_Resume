from app.parser import extract_text_from_pdf
from pathlib import Path
import json
import time

# Load config
OUTPUT_LABELS = Path("output/ground_truth_labels.json")
RESUME_DIR = Path("data/resumes_test")
JD_FILE = Path("data/job_descriptions/sample_jd.txt")


def extract_skills_from_text(text):
    """Extract skills from the 'Skills:' line in synthetic PDFs."""
    for line in text.splitlines():
        if "skills:" in line.lower():
            parts = line.split(":")
            if len(parts) > 1:
                return parts[1].strip()
    return None


def label_resumes():
    labels = {}

    print("\n🔎 Loading resumes...")
    resume_files = sorted(RESUME_DIR.glob("*.pdf"))

    for file in resume_files:
        print("\n" + "-" * 30)
        print(f"📄 File: {file.name}")

        resume_text = extract_text_from_pdf(file)
        skills_extracted = extract_skills_from_text(resume_text)

        # Show info
        print(f"🛠️ Skills: {skills_extracted if skills_extracted else 'No skills found'}")

        # Ask user input
        while True:
            response = input("Is this resume relevant to the Job Description? (y/n): ").strip().lower()
            if response in ("y", "n"):
                labels[file.name] = 1 if response == "y" else 0
                break
            else:
                print("Invalid input. Please type 'y' or 'n'.")

    # Save labels
    OUTPUT_LABELS.parent.mkdir(exist_ok=True)
    with open(OUTPUT_LABELS, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"\nLabeling completed! Saved to {OUTPUT_LABELS}")


if __name__ == "__main__":
    start = time.time()
    label_resumes()
    end = time.time()
    print(f"\nTime taken: {round(end - start, 2)} seconds")
