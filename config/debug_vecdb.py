from pathlib import Path
from vector_store import carregar_banco_vetorial

PASTA_INDEX = Path("faiss_index")


def main():
    """Run a direct FAISS diagnostic lookup for a sample query.

    :raises SystemExit: If the FAISS index is missing.
    """
    print("--- DIAGNÓSTICO DO BANCO VETORIAL FAISS ---")

    if not PASTA_INDEX.exists():
        print(f"[ERROR] O índice não existe em '{PASTA_INDEX.resolve()}'. Rode o 'ingestao.py' primeiro.")
        raise SystemExit(1)

    db = carregar_banco_vetorial()
    pergunta_busca = "Segundo o Provimento 74, o que é exigido sobre backup em nuvem nos cartórios?"
    print(f"\nRealizando busca de teste por: '{pergunta_busca}'")

    documentos = db.similarity_search(pergunta_busca, k=4)

    print(f"\nResultados encontrados no FAISS: {len(documentos)}")
    print("-" * 50)

    for i, doc in enumerate(documentos):
        id_chunk = doc.metadata.get("id_chunk", "ID_DESCONHECIDO")
        fonte = doc.metadata.get("fonte", "Desconhecida")
        print(f"[{i + 1}] ID: {id_chunk} | Fonte: {fonte}")
        print(f"Trecho: {doc.page_content[:250]}...")
        print("-" * 50)


if __name__ == "__main__":
    main()