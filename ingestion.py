from pathlib import Path
import json
from langchain_core.documents import Document
from config.embeddings import obter_pipeline_embeddings
from langchain_community.vectorstores import FAISS

PASTA_CLEANSED = Path("data/cleansed")
PASTA_INDEX = Path("faiss_index")


def carregar_documentos():
    """Load cleaned JSON documents and convert them into LangChain document objects.

    :returns: A list of prepared document chunks.
    :rtype: list[Document]
    """
    documentos = []
    
    if not PASTA_CLEANSED.exists():
        raise FileNotFoundError(f"A pasta de entrada não existe: {PASTA_CLEANSED.resolve()}")
        
    for arquivo in PASTA_CLEANSED.glob("*.json"):
        print(f"Lendo e preparando: {arquivo.name}")
        
        with open(arquivo, 'r', encoding='utf-8') as f:
            dados = json.load(f)
            
            for item in dados:
                metadados_completos = item['metadados'].copy()
                metadados_completos['id_chunk'] = item['id_chunk']
                
                doc = Document(
                    page_content=item['texto'],
                    metadata=metadados_completos
                )
                documentos.append(doc)
    
    return documentos

def main():
    """Run the data ingestion pipeline and build the FAISS index."""
    print("--- Iniciando Ingestão de Dados do RAG ---")
    try:
        docs = carregar_documentos()
        print(f"Total de fragmentos (chunks) preparados: {len(docs)}")

        if len(docs) == 0:
            print("[WARNING] Nenhum documento encontrado. Verifique se executou os scripts de Cleansing.")
            raise SystemExit(1)

        embeddings = obter_pipeline_embeddings()

        print("Gerando embeddings e criando índice FAISS (processando na GPU)...")
        db = FAISS.from_documents(docs, embeddings)

        PASTA_INDEX.mkdir(parents=True, exist_ok=True)
        db.save_local(str(PASTA_INDEX))
        print(f"\n[OK] SUCESSO! Banco vetorial atualizado e salvo em: {PASTA_INDEX.resolve()}")

    except Exception as e:
        print(f"[ERROR] Erro durante a ingestão: {e}")


if __name__ == "__main__":
    main()