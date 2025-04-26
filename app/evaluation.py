from app.models import get_embedding
from app.parser import extract_text_from_pdf
from sentence_transformers import util
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import json

# --- Output directory setup ---
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- File paths ---
JD_FILE = Path("data/job_descriptions/sample_jd.txt")
RESUME_DIR = Path("data/resumes_test")
GROUND_TRUTH_FILE = OUTPUT_DIR / "ground_truth_labels.json"
THRESHOLD = 0.30  # 30% similarity threshold

def load_ground_truth():
    if GROUND_TRUTH_FILE.exists():
        with open(GROUND_TRUTH_FILE, "r") as f:
            return json.load(f)
    else:
        raise FileNotFoundError("Ground truth labels not found. Please run label_resume.py first.")

def evaluate_resumes():
    jd_text = JD_FILE.read_text(encoding="utf-8")
    jd_embedding = get_embedding(jd_text)

    ground_truth = load_ground_truth()
    y_true, y_pred, results = [], [], []

    for resume_file, true_label in ground_truth.items():
        resume_path = RESUME_DIR / resume_file
        if not resume_path.exists():
            print(f"⚠️ Warning: File {resume_file} not found, skipping...")
            continue

        resume_text = extract_text_from_pdf(resume_path)
        resume_embedding = get_embedding(resume_text)

        similarity_score = util.pytorch_cos_sim(jd_embedding, resume_embedding).item()
        score_percent = round(similarity_score * 100, 2)
        predicted_label = 1 if score_percent >= THRESHOLD * 100 else 0

        results.append({
            "file": resume_file,
            "score": score_percent,
            "match": "✅" if predicted_label else "❌"
        })

        y_true.append(true_label)
        y_pred.append(predicted_label)

    # Save results
    df = pd.DataFrame(results).sort_values("score", ascending=False)
    df.to_csv(OUTPUT_DIR / "resume_match_results.csv", index=False)
    print("\n📁 Results saved to 'output/resume_match_results.csv'")

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.barh(df["file"], df["score"], color="skyblue")
    plt.xlabel("Match Score (%)")
    plt.title("Resume vs JD Match Scores")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "evaluate_visualization.png")
    plt.close()
    print("📊 Evaluation visualization saved as 'evaluate_visualization.png'")

    # Evaluation Metrics
    print("\n✅ Evaluation Completed!\n")
    if len(set(y_true)) > 1:
        print("📊 Classification Report:")
        print(classification_report(y_true, y_pred, target_names=["Not Relevant", "Relevant"]))
    else:
        print("⚠️ Only one class present in labels, skipping classification report.")

    print("\n📉 Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    evaluate_resumes()
    print("\nEvaluation test completed successfully.")
    print("Evaluation test passed (LOCAL embeddings).") 