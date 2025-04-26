# AI-Powered Resume Screener with LLM Ranking, Bias Evaluation, and Streamlit Dashboard

##  Project Overview
An **AI-powered resume screening system** that matches resumes to job descriptions using Local (Sentence Transformers) or OpenAI models, with built-in fairness bias checks, evaluation metrics, and an interactive **Streamlit web app**!

---

##  Project Structure
```
├── app/
│   ├── __init__.py
│   ├── parser.py
│   ├── models.py
│   ├── matcher.py
│   ├── bias_checker.py
│   ├── batch_bias_checker.py
│   ├── evaluation.py
│   ├── screen_resumes.py
│   ├── label_resume.py
│   └── utils.py
│
├── tests/
│   ├── test_parser.py
│   ├── test_models.py
│   ├── test_matcher.py
│   ├── test_bias_checker.py
│   ├── test_batch_bias_checker.py
│   ├── test_evaluation.py
│   └── test_screen_resumes.py
│
├── data/
│   ├── resumes/
│   ├── resumes_test/
│   └── job_descriptions/
│
├── output/     #  All output (CSV, plots, JSON) is neatly stored here
│
├── frontend/
│   └── streamlit_app.py
│
├── config.py
├── requirements.txt
├── README.md
├── 
└── generate_test_resumes.py
```

---

##  Key Features
- **Resume Parsing:** Extract name, email, phone, and skills.
- **Resume Matching:** Calculate semantic similarity to JD using embeddings.
- **Batch Resume Screening:** Mass process resumes and shortlists.
- **Bias Checker:** Simulate different names and check for unfair biases.
- **Evaluation Metrics:** Precision, Recall, F1-Score, Confusion Matrix.
- **Ground Truth Labeling:** Label relevance once, auto-evaluate forever.
- **Interactive Streamlit App:** Upload resumes, view scores, visualize results.
- **Local/Cloud Flexible:** Switch easily between Sentence Transformers or OpenAI.
- **Visual Output:** Beautiful bar plots and CSV exports for analysis.

---

##  Tech Stack
- **Python 3.8+**
- **Sentence Transformers** (Hugging Face)
- **OpenAI API** (optional)
- **spaCy** (NLP)
- **scikit-learn** (Evaluation Metrics)
- **Streamlit** (Web App)
- **PyMuPDF (fitz)** (PDF Reading)
- **FPDF** (Generating fake resumes)

---

##  Setup Instructions
```bash
# Clone the project
$ git clone https://github.com/
$ cd AI-Powered-Resume-Screener

# Create and activate virtual environment (optional but recommended)
$ python -m venv venv
$ source venv/bin/activate  # (Linux/Mac)
$ venv\Scripts\activate    # (Windows)

# Install dependencies
$ pip install -r requirements.txt

# Set your config.py (Local or OpenAI)
```

---

##  Running Streamlit Frontend
```bash
$ streamlit run frontend/streamlit_app.py
```

---

##  Example Output
- Resume Match Scores
- Bias Check Visualizations
- Confusion Matrix
- Precision/Recall Reports
- Streamlit Interactive Web App!

---

##  Future Enhancements
- Multi-job matching support
- Advanced Bias mitigation using adversarial debiasing
- Admin Dashboard for HR teams

---

##  Acknowledgements
Thanks to HuggingFace, OpenAI, Streamlit, and scikit-learn communities.

---