import os
import re
import json
import fitz  # PyMuPDF
import torch
from sentence_transformers import SentenceTransformer, util

# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================
PASTA_ENTRADA = os.path.join("data", "raw")
PASTA_SAIDA = os.path.join("data", "cleansed")

ARQUIVOS_ALVO = [
    "prov134.pdf",
    "guia_seg_da_info_agentes_tratamento.pdf",
    "guia_tratamento_dados_pessoas_poder_publico.pdf"
]

# Em vez de palavras-chave, criamos uma "Âncora Semântica". 
# O modelo vai procurar textos cujo significado se pareça com o significado desta frase:
TEXTO_ANCORA = (
    "Regras e orientações sobre segurança da informação, proteção de dados pessoais, "
    "privacidade, confidencialidade, controle de acesso, rastreabilidade de logs, "
    "prevenção contra vazamento de dados, criptografia e adequação à LGPD."
)

# Limiar de Similaridade (Ajuste se necessário. 0.30 a 0.40 costuma ser ideal para MPNet)
LIMIAR_SIMILARIDADE = 0.4

# ==========================================
# 2. METADADOS COMPLETOS
# ==========================================
METADADOS_ARQUIVOS = {
    "prov134.pdf": {
        "fonte": "Corregedoria Nacional de Justiça (CNJ)",
        "data": "2022",
        "tipo": "Provimento",
        "tema": "Adequação das Serventias Extrajudiciais à LGPD",
        "status": "legítimo",
        "finalidade_experimento": "contexto_base"
    },
    "guia_seg_da_info_agentes_tratamento.pdf": {
        "fonte": "Autoridade Nacional de Proteção de Dados (ANPD)",
        "data": "2021",
        "tipo": "Guia Orientativo",
        "tema": "Segurança da Informação para Agentes de Tratamento",
        "status": "legítimo",
        "finalidade_experimento": "contexto_base"
    },
    "guia_tratamento_dados_pessoas_poder_publico.pdf": {
        "fonte": "Autoridade Nacional de Proteção de Dados (ANPD)",
        "data": "2022",
        "tipo": "Guia Orientativo",
        "tema": "Tratamento de Dados pelo Poder Público",
        "status": "legítimo",
        "finalidade_experimento": "contexto_base"
    }
}

# ==========================================
# 3. FUNÇÕES DE LIMPEZA
# ==========================================
def extrair_texto_pdf(caminho_arquivo):
    doc = fitz.open(caminho_arquivo)
    texto_completo = ""
    for pagina in doc:
        texto_completo += pagina.get_text("text") + "\n"
    doc.close()
    return texto_completo

def limpar_e_reconstruir_paragrafos(texto_bruto):
    """
    Como o PDF perdeu a formatação, vamos transformar tudo em texto corrido puro
    e deixar a função de chunking decidir onde cortar.
    """
    # Remove números de página isolados
    texto = re.sub(r'(?m)^\s*\d+\s*$', '', texto_bruto)
    
    # Substitui TODAS as quebras de linha por espaço
    texto = texto.replace('\n', ' ')
    
    # Remove espaços em branco duplos ou múltiplos criados pelo passo anterior
    texto = re.sub(r'\s{2,}', ' ', texto)
    
    return texto.strip()

def segmentar_em_chunks(texto_limpo, tamanho_maximo=1000, tamanho_overlap=150):
    """
    Corta o texto baseado em quantidade de caracteres (Max: 1000), 
    usando um overlap (sobreposição) para não perder o contexto da frase.
    Esta é a técnica mais resiliente para RAG.
    """
    palavras = texto_limpo.split(' ')
    chunks = []
    chunk_atual = []
    tamanho_atual = 0

    for palavra in palavras:
        if not palavra: continue
        
        tamanho_palavra = len(palavra) + 1 # +1 para contar o espaço em branco

        if tamanho_atual + tamanho_palavra > tamanho_maximo:
            # 1. O bloco atingiu o limite. Salvamos ele.
            chunks.append(" ".join(chunk_atual))
            
            # 2. Lógica do Overlap (pega as últimas palavras do bloco atual para iniciar o próximo)
            overlap_words = []
            overlap_len = 0
            for w in reversed(chunk_atual):
                overlap_len += len(w) + 1
                if overlap_len > tamanho_overlap:
                    break
                overlap_words.insert(0, w)
            
            # 3. Começa o novo bloco com o overlap + a palavra atual
            chunk_atual = overlap_words + [palavra]
            tamanho_atual = sum(len(w) + 1 for w in chunk_atual)
        else:
            chunk_atual.append(palavra)
            tamanho_atual += tamanho_palavra
            
    # Adiciona o que sobrou no final do arquivo
    if chunk_atual:
        chunks.append(" ".join(chunk_atual))
        
    return chunks

# ==========================================
# 4. FILTRAGEM SEMÂNTICA (IA)
# ==========================================
def filtrar_relevantes_ia(chunks, modelo, embedding_ancora):
    """
    Usa a IA para converter o chunk em vetor numérico e medir a 
    distância semântica em relação ao nosso tema alvo.
    """
    chunks_uteis = []
    
    # Processa em lotes para usar a GPU de forma super rápida
    # Convert vectors (tensors) to CPU memory afterwards to prevent VRAM leak
    embeddings_chunks = modelo.encode(chunks, convert_to_tensor=True)
    
    # Calcula a similaridade de Cosseno de todos os chunks contra a Âncora de uma vez só
    similaridades = util.cos_sim(embeddings_chunks, embedding_ancora)
    
    for i, chunk in enumerate(chunks):
        score = similaridades[i][0].item() # Pega o valor numérico
        if score >= LIMIAR_SIMILARIDADE:
            chunks_uteis.append(chunk)
            
    return chunks_uteis

# ==========================================
# 5. EXECUÇÃO PRINCIPAL
# ==========================================
def processar_pdfs():
    print("Iniciando pipeline com Filtragem Semântica (GPU)...\n")
    
    # 1. Verifica se a GPU está disponível e carrega o Modelo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware detectado: {device.upper()}")
    print("Carregando modelo de linguagem na memória...")
    
    modelo = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2', device=device)
    
    # Gera o vetor numérico (embedding) do nosso escopo temático
    embedding_ancora = modelo.encode(TEXTO_ANCORA, convert_to_tensor=True)

    for arquivo in ARQUIVOS_ALVO:
        caminho_arquivo = os.path.join(PASTA_ENTRADA, arquivo)
        if not os.path.exists(caminho_arquivo):
            print(f"⚠️ AVISO: '{arquivo}' não encontrado. Pulando...")
            continue

        metadados = METADADOS_ARQUIVOS[arquivo]

        print(f"\n[Processando {arquivo}]")
        texto_bruto = extrair_texto_pdf(caminho_arquivo)
        texto_limpo = limpar_e_reconstruir_paragrafos(texto_bruto)
        chunks_totais = segmentar_em_chunks(texto_limpo)
        
        print(f"  - Avaliando semântica de {len(chunks_totais)} blocos de texto...")
        chunks_relevantes = filtrar_relevantes_ia(chunks_totais, modelo, embedding_ancora)

        dados_documento = []
        prefixo_id = arquivo.replace('.pdf', '')
        
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

if __name__ == "__main__":
    processar_pdfs()