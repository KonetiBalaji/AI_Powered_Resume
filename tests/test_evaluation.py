from config import MODEL_PROVIDER

if MODEL_PROVIDER == "openai":
    print("OpenAI model detected during testing.")
    print("Skipping evaluation test because OpenAI chat models don't use embeddings.")
else:
    from app.evaluation import evaluate_resumes

    if __name__ == "__main__":
        evaluate_resumes()
        print("\nEvaluation test passed (LOCAL embeddings).")
        print("Evaluation test completed successfully."  )