import requests
from config.pinecone import index
from config.settings import settings

def embed_text_pinecone(text):
    """Generate embedding from text using Pinecone's model and store it."""
    # Define the Pinecone embedding API URL
    url = "https://evey-db-ta9ip7v.svc.aped-4627-b74a.pinecone.io/vectors/embedding"

    # Pinecone embedding API headers
    headers = {
        "Authorization": f"Bearer {settings.PINECONE_API_KEY}",
        "Content-Type": "application/json",
    }

    # Payload data for embedding generation
    data = {
        "model": "llama-text-embed-v2",  # Specify the embedding model
        "input": [text],                 # Input text for which we need to generate embeddings
    }

    # Debug: print request info
    print(f"Making request to Pinecone with data: {data}")
    print(f"Request URL: {url}")
    print(f"Request headers: {headers}")

    # Send request to Pinecone to generate embeddings
    response = requests.post(url, json=data, headers=headers)

    # Check the response status
    if response.status_code != 200:
        print("Error generating embedding:", response.text)
        return None

    # Extract the embedding from the response
    embedding = response.json().get("data", [{}])[0].get("embedding")
    if not embedding:
        print("Embedding generation failed!")
        return None

    # Upsert the generated embedding to Pinecone
    upsert_response = index.upsert([(text, embedding)])
    print("Upsert response from Pinecone:", upsert_response)

    return embedding

if __name__ == "__main__":
    sample_text = "Can you tell me about events?"
    emb_via_pinecone = embed_text_pinecone(sample_text)
    print("Embedding (Pinecone API):", emb_via_pinecone)
