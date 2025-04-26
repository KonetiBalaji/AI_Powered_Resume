from config import MODEL_PROVIDER
from app.models import get_embedding, extract_skills_with_model
import torch

sample_text = """
Balaji Koneti is an experienced software engineer with 3+ years in Python, Java, and SQL development.
He has built scalable AI applications using TensorFlow, PyTorch, and NLP frameworks. 
Worked at Infosys and contributed to projects in AWS and Dockerized microservices.
"""

print("MODEL_PROVIDER:", MODEL_PROVIDER)

# Test skill extraction (GPT or fallback)
print("\nExtracted Skills / Summary:")
skills = extract_skills_with_model(sample_text)
print(skills)

# Test embedding generation
print("\nGenerating Embedding:")
embedding = get_embedding(sample_text)

if isinstance(embedding, torch.Tensor):
    print("Embedding shape:", embedding.shape)
else:
    print("Embedding is not a torch Tensor! Something wrong.")
