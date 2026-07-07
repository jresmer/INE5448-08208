import torch
from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings_pipeline():
    """Create and return a Hugging Face embedding pipeline.

    :returns: An embedding model configured for the available device.
    :rtype: HuggingFaceEmbeddings
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[{__name__}] Inicializando Embeddings no dispositivo: {device.upper()}")

    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}

    embeddings_pipeline = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    return embeddings_pipeline
