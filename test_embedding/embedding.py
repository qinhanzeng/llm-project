import openai
import os

client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

response = client.embeddings.create(
    input="Machine Learning is a subfield of AI",
    model="text-embedding-3-small"
)

embedding = response.data[0].embedding

print(f"vector length: {len(embedding)}")
print(f"top 5 values: {embedding[:5]}")
print(f"value range: {min(embedding):.4f} to {max(embedding):.4f}")

