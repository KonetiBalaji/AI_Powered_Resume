from fpdf import FPDF
from pathlib import Path
import random

# Output folder
output_dir = Path("data/resumes_test")
output_dir.mkdir(parents=True, exist_ok=True)

# Sample Names and Skills (expandable)
first_names = ["Alice", "Bob", "Cathy", "David", "Emma", "Frank", "Grace", "Harry", "Isabel", "John",
               "Kelly", "Leo", "Mona", "Nathan", "Olivia", "Paul", "Queen", "Ryan", "Sophia", "Tom",
               "Uma", "Victor", "Wendy", "Xander", "Yara", "Zane"]

last_names = ["Johnson", "Smith", "Zhang", "Lee", "Davis", "Thomas", "Wong", "Wilson", "Moore", "Doe",
              "Taylor", "Brown", "Anderson", "White", "Martin", "Clark", "Lewis", "Walker", "Hall", "Allen",
              "Young", "King", "Wright", "Scott", "Green", "Baker"]

skills_positive = ["Python", "SQL", "Machine Learning", "Deep Learning", "NLP", "Data Engineering", "Spark", "TensorFlow", "PyTorch", "AWS"]
skills_negative = ["Manual Testing", "JIRA", "Tech Support", "Customer Service", "Frontend Dev", "UI/UX", "Networking", "Firewalls", "MS Excel", "Data Entry"]

resumes = {}

# Generate 50 fake resumes
for i in range(1, 51):
    name = f"{random.choice(first_names)} {random.choice(last_names)}"
    if i % 2 == 0:
        skills = ", ".join(random.sample(skills_positive, 4))
        label = 1  # Relevant
    else:
        skills = ", ".join(random.sample(skills_negative, 4))
        label = 0  # Not relevant

    filename = f"resume_{i}.pdf"
    resumes[filename] = (name, skills, label)

    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Name: {name}", ln=True)
    pdf.cell(200, 10, txt=f"Skills: {skills}", ln=True)
    pdf.output(str(output_dir / filename))

# Create GROUND_TRUTH dictionary
labels = {fname: label for fname, (_, _, label) in resumes.items()}

print("50 fake resumes generated.")
print("\nSample GROUND_TRUTH dictionary:")
print(labels)
