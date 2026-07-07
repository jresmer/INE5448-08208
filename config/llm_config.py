from pathlib import Path
from langchain_community.llms import Ollama

CAMINHO_CONFIG = Path("config")
ARQUIVO_ENV = CAMINHO_CONFIG / ".env"


def get_llm(local: bool = True):
    """Return the configured language model instance.

    :param local: Whether to use the local Ollama deployment.
    :type local: bool
    :returns: A configured LLM instance.
    :rtype: Ollama
    """
    if local:
        print(f"[{__name__}] Inicializando LLM Local via Ollama...")
        return Ollama(
            model="gemma3:4b",
            temperature=0.1,
        )

    raise NotImplementedError("Remote LLM configuration is not enabled in this module.")


def main():
    """Run a smoke test for the local LLM configuration."""
    print("Testando configuração do LLM...")
    try:
        get_llm(local=True)
        print("[OK] Instância do LLM configurada com sucesso!")
    except Exception as exc:
        print(f"[ERROR] Erro na configuração do LLM: {exc}")


if __name__ == "__main__":
    main()