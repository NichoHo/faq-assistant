import os
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
        return [self._to_vector(self._client.feature_extraction(text)) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._to_vector(self._client.feature_extraction(text))


def get_embeddings():
    """Return embeddings model, using HF Inference API on Vercel to avoid bundling torch."""
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    hf_token = os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    model_name = os.environ.get(
        "EMBEDDINGS_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    if os.environ.get("VERCEL") == "1" or hf_token:
        if not hf_token:
            raise RuntimeError(
                "Set HF_API_TOKEN (or HUGGINGFACEHUB_API_TOKEN) in Vercel project settings "
                "for embedding queries. Create a free token at https://huggingface.co/settings/tokens"
            )

        os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", hf_token)

        _embeddings = HuggingFaceInferenceEmbeddings(
            model=model_name,
            api_token=hf_token,
        )
        return _embeddings

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise RuntimeError(
            "Local embeddings require sentence-transformers and torch. "
            "Install them with: pip install -r requirements-local.txt "
            "Or set HF_API_TOKEN to use Hugging Face Inference API."
        ) from exc

    _embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
    )
    return _embeddings
