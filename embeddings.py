import os
import time
from typing import List
from langchain_core.embeddings import Embeddings

_embeddings = None

class HuggingFaceInferenceEmbeddings(Embeddings):

    def __init__(self, model: str, api_token: str):
        from huggingface_hub import InferenceClient
        self.model = model
        self._client = InferenceClient(model=model, token=api_token)

    def _to_vector(self, result) -> List[float]:
        import numpy as np
        arr = np.asarray(result, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[0] if arr.shape[0] == 1 else arr.mean(axis=0)
        return arr.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # The Goldilocks Fix: Process in batches of 15 to avoid timeouts and payload limits
        batch_size = 15
        all_embeddings = []
        
        print(f"📦 Processing {len(texts)} chunks in batches of {batch_size}...")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                results = self._client.feature_extraction(batch)
                all_embeddings.extend([self._to_vector(res) for res in results])
                
                # Tiny pause to respect Hugging Face free tier rate limits
                time.sleep(0.2)
            except Exception as e:
                print(f"❌ Error processing batch {i} to {i+batch_size}: {str(e)}")
                raise e
                
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._to_vector(self._client.feature_extraction(text))


def get_embeddings():
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    hf_token = os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    model_name = os.environ.get(
        "EMBEDDINGS_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    if os.environ.get("VERCEL") == "1" or hf_token:
        os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", hf_token)
        _embeddings = HuggingFaceInferenceEmbeddings(
            model=model_name,
            api_token=hf_token,
        )
        return _embeddings

    from langchain_huggingface import HuggingFaceEmbeddings
    _embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
    )
    return _embeddings