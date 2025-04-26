# tests/test_screen_resumes.py

import os
import pandas as pd
from pathlib import Path

# Import the function you want to test
from app.screen_resumes import screen_resumes

OUTPUT_CSV = Path("output/real_resume_screening_results.csv")

def test_screen_resumes_creates_csv():
    # Remove output file if exists (fresh test)
    if OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()

    # Run the screen_resumes function
    screen_resumes()

    # Check if output file was created
    assert OUTPUT_CSV.exists(), "CSV file was not created."

    # Check basic content
    df = pd.read_csv(OUTPUT_CSV)
    assert not df.empty, "CSV file is empty."
    assert "file" in df.columns, "'file' column missing."
    assert "score" in df.columns, "'score' column missing."
    assert "shortlisted" in df.columns, "'shortlisted' column missing."

    print("✅ test_screen_resumes_creates_csv passed.")

if __name__ == "__main__":
    test_screen_resumes_creates_csv()
