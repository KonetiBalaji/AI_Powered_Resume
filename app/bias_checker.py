import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
# Add this right after your imports
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)  # Makes sure the folder exists

from app.models import get_embedding
from app.parser import extract_text_from_pdf
from sentence_transformers import util
from pathlib import Path

# Simulated names from diverse backgrounds
SIMULATED_NAMES = [
    "Emily Johnson", "Aisha Khan", "Wei Zhang", "Carlos Gonzalez", "Liam Smith",
    "Lakshmi Narayanan", "Ahmed El-Sayed", "Hiroshi Tanaka", "Olga Ivanova", "Jamal Washington"
]

def inject_name(text, name):
    lines = text.strip().splitlines()
    lines[0] = name  # Replace first line with new name
    return "\n".join(lines)

def simulate_bias_check(jd_path, resume_path):
    jd_text = Path(jd_path).read_text(encoding="utf-8")
    jd_embedding = get_embedding(jd_text)

    base_resume_text = extract_text_from_pdf(resume_path)
    results = []

    for name in SIMULATED_NAMES:
        modified_resume = inject_name(base_resume_text, name)
        resume_embedding = get_embedding(modified_resume)
        score = util.pytorch_cos_sim(jd_embedding, resume_embedding).item()

        results.append({"name": name, "score": round(score * 100, 2)})

    return sorted(results, key=lambda x: x["score"], reverse=True)

# Run from CLI
# Everything same till...

if __name__ == "__main__":
    jd_file = "data/job_descriptions/sample_jd.txt"
    resume_file = Path("data/resumes/sample_resume.pdf")

    print("Running Bias Checker...\n")
    results = simulate_bias_check(jd_file, resume_file)

    for r in results:
        print(f"{r['name']:25} — Score: {r['score']}%")

    # Now move SAVE and PLOT inside here 
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "bias_check_results.csv", index=False)
    print("\nResults saved to 'bias_check_results.csv'")

    # plot results
    names = [r["name"] for r in results]
    scores = [r["score"] for r in results]

    plt.figure(figsize=(12, 6))
    plt.barh(names, scores, color="skyblue")
    plt.xlabel("Match Score (%)")
    plt.title("Bias Check – Resume Score by Simulated Name")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bias_visualization.png")
    plt.show()
    plt.close()  # Free memory
    print("\nVisualization saved to 'bias_visualization.png'")
