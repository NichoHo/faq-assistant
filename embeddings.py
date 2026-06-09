import os

_embeddings = None


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

        from langchain_huggingface import HuggingFaceEndpointEmbeddings

        _embeddings = HuggingFaceEndpointEmbeddings(
            model=model_name,
            huggingfacehub_api_token=hf_token,
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
