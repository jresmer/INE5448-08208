from pathlib import Path
from config.vector_store import carregar_banco_vetorial

INDEXING_PATH = Path("faiss_index")


def get_retriever(k_documents: int = 4):
    """Create a similarity retriever for the FAISS index.

    :param k_documents: Number of documents to retrieve.
    :type k_documents: int
    :returns: A retriever configured for similarity search.
    :rtype: object
    """
    print(f"[{__name__}] Configurando o Retriever do RAG...")

    if not INDEXING_PATH.exists() or not (INDEXING_PATH / "index.faiss").exists():
        raise FileNotFoundError(
            f"O índice FAISS não foi localizado no caminho esperado: {INDEXING_PATH.resolve()}\n"
            "Por favor, execute o script de indexação primeiro."
        )

    db = carregar_banco_vetorial()

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k_documents})

    print(f"[{__name__}] [OK] Retriever pronto para busca por similaridade (K={k_documents}).")
    return retriever
