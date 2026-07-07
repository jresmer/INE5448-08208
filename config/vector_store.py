from pathlib import Path
from langchain_community.vectorstores import FAISS
from config.embeddings import get_embeddings_pipeline

PASTA_INDEX = Path("faiss_index")


def salvar_banco_vetorial(documentos):
    """Create and save a FAISS vector store from a list of documents.

    :param documentos: Documents to be indexed.
    :type documentos: list
    :returns: The created FAISS vector store.
    :rtype: FAISS
    """
    print(f"[{__name__}] Iniciando indexação de {len(documentos)} documentos...")

    embeddings = get_embeddings_pipeline()

    db = FAISS.from_documents(documentos, embeddings)

    PASTA_INDEX.mkdir(parents=True, exist_ok=True)

    db.save_local(str(PASTA_INDEX))
    print(f"[{__name__}] [OK] Banco vetorial indexado e salvo em: {PASTA_INDEX.resolve()}")
    return db


def carregar_banco_vetorial():
    """Load the persisted FAISS vector store from disk.

    :returns: The loaded FAISS vector store.
    :rtype: FAISS
    """
    if not PASTA_INDEX.exists() or not PASTA_INDEX.is_dir():
        raise FileNotFoundError(
            f"O diretório do índice FAISS não existe ou é inválido: {PASTA_INDEX.resolve()}"
        )

    arquivo_faiss = PASTA_INDEX / "index.faiss"
    arquivo_pkl = PASTA_INDEX / "index.pkl"

    if not arquivo_faiss.exists() or not arquivo_pkl.exists():
        raise FileNotFoundError(
            f"Índice corrompido ou incompleto na pasta: {PASTA_INDEX.resolve()}\n"
            f"Certifique-se de que os arquivos '{arquivo_faiss.name}' e '{arquivo_pkl.name}' estão presentes."
        )

    embeddings = get_embeddings_pipeline()

    db = FAISS.load_local(str(PASTA_INDEX), embeddings, allow_dangerous_deserialization=True)
    return db
