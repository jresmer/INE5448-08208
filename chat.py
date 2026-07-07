import sys
from pathlib import Path
from config.prompt import get_rag_chain

INDEX_PATH = Path("faiss_index")


def start_chat():
    """Start an interactive chat session backed by the RAG pipeline.

    :raises SystemExit: If the FAISS index is missing or the agent cannot initialize.
    """
    if not INDEX_PATH.exists() or not (INDEX_PATH / "index.faiss").exists():
        print(f"[ERROR] O índice vetorial não foi encontrado em: {INDEX_PATH.resolve()}")
        print("Por favor, execute 'python ingestao.py' antes de iniciar o chat.")
        sys.exit(1)

    print("Carregando o Agente RAG na memória (isso pode levar alguns segundos)...")
    try:
        rag_chain = get_rag_chain()
    except Exception as e:
        print(f"[ERROR] Falha crítica ao inicializar o agente: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("[AGENT] AGENTE JURÍDICO RAG - CHAT INTERATIVO")
    print("Pergunte sobre LGPD, Provimento 213, Provimento 74 ou Resolução 396.")
    print("Digite 'sair' ou 'exit' para encerrar a sessão.")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("\n[USER] Você: ").strip()

            if not question:
                continue

            if question.lower() in ["sair", "exit", "quit", "q"]:
                print("\nEncerrando sessão de chat. Até logo!")
                break

            print("[AGENT] Agente pensando...")

            response = rag_chain.invoke(question)

            print(f"\n[AGENT] Agente:\n{response}")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n\nSessão encerrada de forma segura pelo usuário. Até logo!")
            break
        except Exception as e:
            print(f"\n[ERROR] Erro durante o processamento da dúvida: {e}")

if __name__ == "__main__":
    start_chat()