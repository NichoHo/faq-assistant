import os

_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    # Use Pinecone's built-in serverless embeddings instead of Hugging Face
    from langchain_pinecone import PineconeEmbeddings
    
    _embeddings = PineconeEmbeddings(
        model="multilingual-e5-large", 
        pinecone_api_key=os.environ.get("PINECONE_API_KEY")
    )
    return _embeddings