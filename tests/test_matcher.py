# tests/test_matcher.py
from app.matcher import match_resumes_to_jd

def test_matching_process():
    jd_file = "data/job_descriptions/sample_jd.txt"
    resumes_folder = "data/resumes"

    matches = match_resumes_to_jd(jd_file, resumes_folder)

    assert isinstance(matches, list), "Output should be a list"
    assert len(matches) > 0, "There should be at least one matched resume"
    assert "score" in matches[0], "Each match should contain a 'score'"
    assert "name" in matches[0], "Each match should contain a 'name'"
    assert "email" in matches[0], "Each match should contain an 'email'"
    assert "file" in matches[0], "Each match should contain a 'file'"

    print("\n test_matching_process passed.")

if __name__ == "__main__":
    test_matching_process()
