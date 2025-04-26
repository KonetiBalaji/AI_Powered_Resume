# tests/test_bias_checker.py

from app.bias_checker import simulate_bias_check
from pathlib import Path

def test_simulate_bias_check():
    jd_file = "data/job_descriptions/sample_jd.txt"
    resume_file = Path("data/resumes/sample_resume.pdf")

    results = simulate_bias_check(jd_file, resume_file)

    assert isinstance(results, list), "Result should be a list"
    assert all("name" in r and "score" in r for r in results), "Each result should have 'name' and 'score'"
    assert all(isinstance(r["score"], float) for r in results), "Scores must be floats"
    assert len(results) == 10, "There should be 10 simulated names checked"

    print("\ntest_simulate_bias_check passed.")

if __name__ == "__main__":
    test_simulate_bias_check()
