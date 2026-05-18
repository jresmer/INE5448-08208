import os
import re
import json
from bs4 import BeautifulSoup
import torch
from sentence_transformers import SentenceTransformer, util

# ==========================================
# 1. CONFIGURAÇÕES DE DIRETÓRIOS E ARQUIVOS
# ==========================================
PASTA_ENTRADA = os.path.join("data", "raw")
PASTA_SAIDA = os.path.join("data", "cleansed")

ARQUIVOS_ALVO = ["res396.html", "prov213.html", "prov74.html", "lgpd.html"]

# ==========================================
# 2. CONFIGURAÇÕES DA IA SEMÂNTICA (GPU)
# ==========================================
# Âncora Semântica: A IA buscará trechos cujo significado se aproxime desta frase.
TEXTO_ANCORA = (
    "Regras e orientações sobre segurança da informação, proteção de dados pessoais, "
    "privacidade, confidencialidade, controle de acesso, rastreabilidade de logs, "
    "prevenção contra vazamento de dados, criptografia e adequação à LGPD."
)

# Limiar de Similaridade (Entre 0.30 e 0.40 é ideal)
LIMIAR_SIMILARIDADE = 0.4 

# ==========================================
# 3. METADADOS COMPLETOS (Plano de Trabalho, pág. 4)
# ==========================================
METADADOS_ARQUIVOS = {
    "prov213.html": {
        "fonte": "Corregedoria Nacional de Justiça (CNJ)",
        "data": "2026",
        "tipo": "Provimento",
        "tema": "Padrões Mínimos de TIC e Proteção de Dados Notariais",
        "status": "legítimo",
        "finalidade_experimento": "contexto_base"
    },
    "prov74.html": {
        "fonte": "Corregedoria Nacional de Justiça (CNJ)",
        "data": "2018",
        "tipo": "Provimento",
        "tema": "Padrões Mínimos de TI e Segurança Notarial",
        "status": "legítimo",
        "finalidade_experimento": "contexto_base"
    },
    "lgpd.html": {
        "fonte": "Presidência da República (Lei 13.709)",
        "data": "2018",
        "tipo": "Lei",
        "tema": "Proteção de Dados Pessoais",
        "status": "legítimo",
        "finalidade_experimento": "contexto_base"
    },
    "res396.html": {
        "fonte": "Conselho Nacional de Justiça (CNJ)",
        "data": "2021",
        "tipo": "Resolução",
        "tema": "Estratégia Nacional de Segurança Cibernética",
        "status": "legítimo",
        "finalidade_experimento": "contexto_base"
    }
}

# ==========================================
# 4. FUNÇÕES DE LIMPEZA E EXTRAÇÃO
# ==========================================
def limpar_html(conteudo_html):
    """Remove tags HTML desnecessárias e extrai o texto em uma linha limpa."""
    soup = BeautifulSoup(conteudo_html, 'html.parser')

    tags_inuteis = ['script', 'style', 'nav', 'header', 'footer', 'aside', 'strike', 'del']
    for tag in soup(tags_inuteis):
        tag.decompose()

    texto_puro = soup.get_text(separator=' ')
    # Remove espaços duplos ou quebras de linha soltas
    texto_limpo = re.sub(r'\s+', ' ', texto_puro).strip()
    return texto_limpo

def segmentar_em_chunks(texto_limpo, tamanho_maximo=1000, tamanho_overlap=150):
    """
    Estratégia Híbrida:
    1. Tenta quebrar por Artigos (Respeita a estrutura jurídica).
    2. Se um Artigo for gigante, divide-o recursivamente mantendo o overlap.
    """
    # 1. Tenta identificar artigos via Regex
    # Procura por "Art. X" ou "Artigo X"
    padrao = r'(?=\b(?:Art\.|Artigo)\s*\d+)'
    blocos_artigos = re.split(padrao, texto_limpo)
    
    chunks_finais = []
    
    for bloco in blocos_artigos:
        bloco = bloco.strip()
        if len(bloco) < 20: continue # Ignora lixo
        
        # 2. Se o bloco cabe no limite, adiciona direto
        if len(bloco) <= tamanho_maximo:
            chunks_finais.append(bloco)
        else:
            # 3. Se é gigante, faz um Sliding Window (quebra recursiva)
            palavras = bloco.split(' ')
            sub_chunk = []
            tamanho_atual = 0
            
            for palavra in palavras:
                tamanho_palavra = len(palavra) + 1
                if tamanho_atual + tamanho_palavra > tamanho_maximo:
                    chunks_finais.append(" ".join(sub_chunk))
                    # Overlap: inicia o próximo com as últimas palavras
                    overlap_words = sub_chunk[-int(tamanho_overlap/6):] # Aproximadamente as últimas palavras
                    sub_chunk = overlap_words + [palavra]
                    tamanho_atual = sum(len(w) + 1 for w in sub_chunk)
                else:
                    sub_chunk.append(palavra)
                    tamanho_atual += tamanho_palavra
            
            if sub_chunk:
                chunks_finais.append(" ".join(sub_chunk))
                
    return chunks_finais

def filtrar_relevantes_ia(chunks, modelo, embedding_ancora):
    """Usa IA para medir a similaridade de cosseno na placa de vídeo."""
    chunks_uteis = []
    
    # Roda tudo na VRAM de uma vez
    embeddings_chunks = modelo.encode(chunks, convert_to_tensor=True)
    similaridades = util.cos_sim(embeddings_chunks, embedding_ancora)
    
    for i, chunk in enumerate(chunks):
        score = similaridades[i][0].item()
        if score >= LIMIAR_SIMILARIDADE:
            chunks_uteis.append(chunk)
            
    return chunks_uteis

# ==========================================
# 5. EXECUÇÃO PRINCIPAL
# ==========================================
def processar_base():
    os.makedirs(PASTA_ENTRADA, exist_ok=True)
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    
    print("Iniciando pipeline HTML com IA Semântica (GPU)...\n")

    # Inicializa VRAM e carrega Modelo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware detectado: {device.upper()}")
    modelo = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2', device=device)
    embedding_ancora = modelo.encode(TEXTO_ANCORA, convert_to_tensor=True)

    for arquivo in ARQUIVOS_ALVO:
        caminho_arquivo = os.path.join(PASTA_ENTRADA, arquivo)
        
        if not os.path.exists(caminho_arquivo):
            print(f"⚠️ AVISO: '{arquivo}' não encontrado. Pulando...")
            continue

        metadados = METADADOS_ARQUIVOS[arquivo]
        print(f"\n[Processando {arquivo}]")

        # Lê lidando com codificações brasileiras antigas
        try:
            with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                html_content = f.read()
        except UnicodeDecodeError:
            with open(caminho_arquivo, 'r', encoding='latin-1') as f:
                html_content = f.read()

        # Executa a esteira (Pipeline)
        texto_limpo = limpar_html(html_content)
        
        print("  - Segmentando em janela deslizante (max 1000 chars com overlap)...")
        chunks_totais = segmentar_em_chunks(texto_limpo)
        
        print(f"  - Analisando semântica de {len(chunks_totais)} chunks na GPU...")
        chunks_relevantes = filtrar_relevantes_ia(chunks_totais, modelo, embedding_ancora)

        dados_documento = []
        prefixo_id = arquivo.split('.')[0]
        
        for i, chunk in enumerate(chunks_relevantes):
            id_chunk = f"{prefixo_id}_chunk_{i+1:03d}"
            registro = {
                "id_chunk": id_chunk,
                "texto": chunk,
                "metadados": metadados
            }
            dados_documento.append(registro)
            
        nome_arquivo_saida = f"{prefixo_id}_cleansed.json"
        caminho_saida = os.path.join(PASTA_SAIDA, nome_arquivo_saida)
        
        with open(caminho_saida, 'w', encoding='utf-8') as f:
            json.dump(dados_documento, f, ensure_ascii=False, indent=4)

        print(f"✅ Concluído!")
        print(f"   -> Retidos: {len(chunks_relevantes)} de {len(chunks_totais)} (Score min: {LIMIAR_SIMILARIDADE})")
        print(f"   -> Salvo em: {caminho_saida}")

if __name__ == "__main__":
    processar_base()