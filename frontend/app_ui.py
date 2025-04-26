# frontend/streamlit_app.py

import streamlit as st
from pathlib import Path
import sys, os, tempfile
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import util

# Add app folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import backend functions
from config import MODEL_PROVIDER
from app.parser import parse_resume, extract_text_from_pdf
from app.models import get_embedding

# Paths
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Streamlit page setup
st.set_page_config(page_title="AI-Powered Resume Screener", layout="wide")
st.title("🤖 AI-Powered Resume Screener")

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", [
    "🏠 Home",
    "📂 Batch Resume Screening",
    "⚡ Quick Match (Upload Resume & JD)",
    "📊 Visualizations",
    "📥 Download Results",
    "⚙️ Settings"
])

# --- Sections ---

if section == "🏠 Home":
    st.header("Welcome!")
    st.markdown("""
        This AI-powered tool allows you to:
        - 🔍 Match resumes to job descriptions
        - 📈 Visualize screening results
        - ✅ Perform quick one-off resume matches
        - 📊 Evaluate model performance

        Supports both **Local SentenceTransformer** and **OpenAI Chat Models**.
    """)

elif section == "📂 Batch Resume Screening":
    st.header("Batch Screening")
    jd_file = Path("data/job_descriptions/sample_jd.txt")
    resumes_dir = Path("data/resumes")

    if jd_file.exists() and resumes_dir.exists():
        st.success("Screening Started...")
        jd_text = jd_file.read_text(encoding="utf-8")
        jd_embedding = get_embedding(jd_text)

        results = []
        progress = st.progress(0)
        for idx, resume_path in enumerate(resumes_dir.glob("*.pdf")):
            resume_text = extract_text_from_pdf(resume_path)
            resume_embedding = get_embedding(resume_text)
            score = util.pytorch_cos_sim(jd_embedding, resume_embedding).item() * 100

            results.append({
                "file": resume_path.name,
                "score": round(score, 2)
            })

            progress.progress((idx + 1) / len(list(resumes_dir.glob("*.pdf"))))

        df = pd.DataFrame(results).sort_values("score", ascending=False)
        df.to_csv(OUTPUT_DIR / "real_resume_screening_results.csv", index=False)

        st.success("Screening Completed!")
        st.dataframe(df)

    else:
        st.error("Job description or resumes not found!")

elif section == "⚡ Quick Match (Upload Resume & JD)":
    st.header("Quick Resume vs JD Matcher")

    with st.form("quick_match_form"):
        resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
        jd_file = st.file_uploader("Upload Job Description (TXT)", type=["txt"])
        submitted = st.form_submit_button("Match Now 🚀")

    if submitted and resume_file and jd_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_resume:
            tmp_resume.write(resume_file.read())
            resume_path = tmp_resume.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_jd:
            tmp_jd.write(jd_file.read())
            jd_path = tmp_jd.name

        parsed = parse_resume(resume_path)
        jd_text = Path(jd_path).read_text(encoding='utf-8')

        try:
            resume_embedding = get_embedding(parsed["raw_text"])
            jd_embedding = get_embedding(jd_text)
            score = util.pytorch_cos_sim(jd_embedding, resume_embedding).item() * 100
        except NotImplementedError as e:
            score = None
            st.warning(f"⚠️ {e}")

        st.subheader("Results")
        if score:
            st.metric("Resume vs JD Match", f"{score:.2f}%")
        else:
            st.error("Unable to compute similarity in current mode.")

        st.subheader("Extracted Resume Info")
        st.write(f"**Name:** {parsed['name']}")
        st.write(f"**Email:** {parsed['email']}")
        st.write(f"**Phone:** {parsed['phone']}")
        skills_display = parsed['skills'] if isinstance(parsed['skills'], str) else ', '.join(parsed['skills'])
        st.write(f"**Skills:** {skills_display}")

    elif submitted:
        st.error("Please upload both a resume and a JD.")

elif section == "📊 Visualizations":
    st.header("Visual Analysis")

    if (OUTPUT_DIR / "real_screening_visualization.png").exists():
        st.image(str(OUTPUT_DIR / "real_screening_visualization.png"), caption="Screening Results")
    else:
        st.info("No screening chart found yet.")

    if (OUTPUT_DIR / "bias_visualization.png").exists():
        st.image(str(OUTPUT_DIR / "bias_visualization.png"), caption="Bias Evaluation Results")
    else:
        st.info("No bias visualization chart available yet.")

elif section == "📥 Download Results":
    st.header("Download Outputs")

    if (OUTPUT_DIR / "real_resume_screening_results.csv").exists():
        with open(OUTPUT_DIR / "real_resume_screening_results.csv", "rb") as file:
            st.download_button(
                label="Download Screening CSV",
                data=file,
                file_name="real_resume_screening_results.csv",
                mime="text/csv"
            )
    else:
        st.info("No screening results to download.")

elif section == "⚙️ Settings":
    st.header("Configuration")
    st.write(f"**Model Provider:** {MODEL_PROVIDER}")
    st.info("To switch models, edit the `config.py` file.")
