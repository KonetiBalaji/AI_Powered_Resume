from config import MODEL_PROVIDER, LOCAL_MODEL_NAME, OPENAI_MODEL_NAME, OPENAI_API_KEY

if MODEL_PROVIDER == "local":
    from sentence_transformers import SentenceTransformer
    import torch

    model = SentenceTransformer(LOCAL_MODEL_NAME)

    def get_embedding(text):
        return model.encode(text, convert_to_tensor=True)

    def extract_skills_with_model(text):
        # Local model doesn't extract skills; fallback
        return f"Model: {LOCAL_MODEL_NAME} — Local embedding used."

elif MODEL_PROVIDER == "openai":
    from openai import OpenAI
    import torch

    client = OpenAI(api_key=OPENAI_API_KEY)

    def get_embedding(text):
        response = client.embeddings.create(
            model="text-embedding-ada-002",  # using OpenAI's best embedding model
            input=text,
        )
        embedding = response.data[0].embedding
        return torch.tensor(embedding).unsqueeze(0)  # convert list to tensor [1, hidden_dim]

    def extract_skills_with_model(text):
        prompt = (
            "You are a helpful assistant that extracts professional skills from resumes.\n"
            "Given the following resume content, list the top 10 most relevant skills for a job in tech:\n\n"
            f"{text}\n\n"
            "Return the skills as a comma-separated list."
        )

        response = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=250
        )

        return response.choices[0].message.content.strip()

else:
    raise ValueError(f"Unsupported MODEL_PROVIDER: {MODEL_PROVIDER}")
