from app.models import get_embedding
from app.parser import parse_resume
from sentence_transformers import util
from pathlib import Path
import torch

def load_job_description(jd_path):
    return Path(jd_path).read_text(encoding="utf-8")

def match_resumes_to_jd(jd_path, resumes_dir):
    jd_text = load_job_description(jd_path)
    jd_embedding = get_embedding(jd_text)

    results = []

    for resume_file in Path(resumes_dir).glob("*.pdf"):
        try:
            resume_data = parse_resume(resume_file)
            resume_text = resume_data["raw_text"]

            # Generate embedding for resume text
            resume_embedding = get_embedding(resume_text)

            # Cosine similarity (works both for torch or list type embeddings)
            if isinstance(jd_embedding, torch.Tensor):
                similarity = util.pytorch_cos_sim(jd_embedding, resume_embedding).item()
            else:
                # fallback, but since we always use torch.tensor(), it's safe
                raise ValueError("Unsupported embedding type")

            results.append({
                "name": resume_data["name"],
                "email": resume_data["email"],
                "score": round(similarity * 100, 2),
                "file": resume_file.name
            })

        except Exception as e:
            print(f"Error processing {resume_file.name}: {e}")

    return sorted(results, key=lambda x: x["score"], reverse=True)

# Local test
if __name__ == "__main__":
    jd_file = "data/job_descriptions/sample_jd.txt"
    resumes_folder = "data/resumes"

    matches = match_resumes_to_jd(jd_file, resumes_folder)
    print("\nMatched Candidates:")
    for match in matches:
        print(f"{match['name']} ({match['email']}) — Score: {match['score']}% — File: {match['file']}")
