from app.models import get_embedding
from app.parser import extract_text_from_pdf
from sentence_transformers import util
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# --- Output directory setup ---
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- File paths ---
JD_FILE = Path("data/job_descriptions/sample_jd.txt")
RESUME_DIR = Path("data/resumes")

# --- Threshold for match indicator ---
THRESHOLD = 0.30  # 30% similarity


def batch_bias_audit():
    jd_text = JD_FILE.read_text(encoding="utf-8")
    jd_embedding = get_embedding(jd_text)

    results = []

    for resume_path in RESUME_DIR.glob("*.pdf"):
        resume_text = extract_text_from_pdf(resume_path)
        resume_embedding = get_embedding(resume_text)

        score = util.pytorch_cos_sim(jd_embedding, resume_embedding).item()
        score_percent = round(score * 100, 2)

        results.append({
            "file": resume_path.name,
            "score": score_percent,
            "match": "✅" if score_percent >= THRESHOLD * 100 else "❌"
        })

    df = pd.DataFrame(results).sort_values("score", ascending=False)
    df.to_csv(OUTPUT_DIR / "real_resume_screening_results.csv", index=False)

    # --- Bar plot ---
    plt.figure(figsize=(12, len(df) * 0.4))
    plt.barh(df["file"], df["score"], color="skyblue")
    plt.xlabel("Match Score (%)")
    plt.title("Real Resume Screening – Score vs JD")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "real_screening_visualization.png")
    plt.close()

    print("\nBatch Bias Screening Completed")
    print("CSV saved to: output/real_resume_screening_results.csv")
    print("Chart saved to: output/real_screening_visualization.png")


if __name__ == "__main__":
    batch_bias_audit()
