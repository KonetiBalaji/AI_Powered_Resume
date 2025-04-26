from pathlib import Path
from app.parser import parse_resume

def test_parse_resume():
    resume_path = Path("data/resumes/sample_resume.pdf")
    parsed = parse_resume(resume_path)

    assert parsed["name"], "Name not extracted"
    assert parsed["email"], "Email not extracted"
    assert parsed["phone"], "Phone not extracted"
    assert parsed["skills"], "Skills not extracted"
    assert "raw_text" in parsed, "raw_text missing"

    print("test_parse_resume passed — All fields extracted properly!")

if __name__ == "__main__":
    test_parse_resume()
