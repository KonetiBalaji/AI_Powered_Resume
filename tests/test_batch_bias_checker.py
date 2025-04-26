# tests/test_batch_bias_checker.py

import os
from pathlib import Path
from app.batch_bias_checker import batch_bias_audit

def test_batch_bias_audit():
    # Run the batch audit function
    batch_bias_audit()

    # Output files
    csv_path = Path("output/real_resume_screening_results.csv")
    chart_path = Path("output/real_screening_visualization.png")

    # Check if CSV file was created
    assert csv_path.exists(), "CSV file not created!"

    # Check if Chart file was created
    assert chart_path.exists(), "Chart file not created!"

    print("\ntest_batch_bias_audit passed.")

if __name__ == "__main__":
    test_batch_bias_audit()
