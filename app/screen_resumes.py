from app.models import get_embedding
from app.parser import extract_text_from_pdf
from sentence_transformers import util
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Setup
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

JD_FILE = "data/job_descriptions/sample_jd.txt"
RESUME_DIR = Path("data/resumes")
THRESHOLD = 0.30  # 30% threshold (consistency with evaluation)

def screen_resumes():
    jd_text = Path(JD_FILE).read_text(encoding="utf-8")
    jd_embedding = get_embedding(jd_text)

    results = []

    for resume_path in RESUME_DIR.glob("*.pdf"):
        resume_text = extract_text_from_pdf(resume_path)
        resume_embedding = get_embedding(resume_text)

        similarity = util.pytorch_cos_sim(jd_embedding, resume_embedding).item()
        score_percent = round(similarity * 100, 2)

        results.append({
            "file": resume_path.name,
            "score": score_percent,
            "shortlisted": "✅" if score_percent >= (THRESHOLD * 100) else "❌"
        })

        print(f"{resume_path.name:25} — Score: {score_percent}% {'✅' if score_percent >= (THRESHOLD * 100) else ''}")

    # Save results
    df = pd.DataFrame(results).sort_values("score", ascending=False)
    df.to_csv(OUTPUT_DIR / "real_resume_screening_results.csv", index=False)

    # Visualization
    plt.figure(figsize=(12, len(df) * 0.4))
    plt.barh(df["file"], df["score"], color="skyblue")
    plt.xlabel("Match Score (%)")
    plt.title("Real Resume Screening – Score vs JD")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "real_screening_visualization.png")

    print("\nResults saved to 'output/real_resume_screening_results.csv'")
    print("Chart saved to 'output/real_screening_visualization.png'")

if __name__ == "__main__":
    screen_resumes()
