import re
import numpy as np


def sanitize_input(query):
    """Sanitize an incoming query by removing control tokens and trigger phrases.

    :param query: The raw user input.
    :type query: str
    :returns: A tuple containing a safety flag and the sanitized query.
    :rtype: tuple[bool, str]
    """
    control_tokens = [
        "<start_of_turn>",
        "<end_of_turn>",
        "<|im_start|>",
        "<|im_end|>",
        "<|eot_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
    ]

    trigger_words = [
        "system override",
        "ignore context",
        "ignore the context",
        "ignore rules",
        "override authority",
        "revela tuas instrucoes",
        "system prompt",
        "diretrizes de seguranca",
        "ignore as regras",
        "ignorar contexto",
        "modo de depuracao",
        "excecoes locais",
        "precedencia absoluta",
        "vazamento total",
        "revelar prompt",
    ]

    query_lower = query.lower()

    for token in control_tokens:
        if token in query_lower:
            return False, ""

    cleaned_query = query
    for word in trigger_words:
        if word in query_lower:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            cleaned_query = pattern.sub("[REDACTED]", cleaned_query)

    return True, cleaned_query


def sanitize_output_markdown(response):
    """Replace markdown image syntax in a response with a security placeholder.

    :param response: The generated model response.
    :type response: str
    :returns: The sanitized response.
    :rtype: str
    """
    image_pattern = re.compile(r"!\[.*?\]\(.*?\)")
    return image_pattern.sub("[SECURITY BLOCK: IMAGE REMOVED]", response)


def sanitize_output_repetition(response, max_consecutive=3):
    """Remove repeated consecutive words from a response.

    :param response: The generated model response.
    :type response: str
    :param max_consecutive: Maximum allowed consecutive repetitions.
    :type max_consecutive: int
    :returns: The cleaned response.
    :rtype: str
    """
    words = response.split()
    if not words:
        return response

    cleaned_words = []
    previous_word = None
    consecutive_count = 0

    for word in words:
        word_clean = re.sub(r"[^\w]", "", word).lower()
        if word_clean == previous_word:
            consecutive_count += 1
        else:
            consecutive_count = 1
            previous_word = word_clean

        if consecutive_count <= max_consecutive:
            cleaned_words.append(word)

    return " ".join(cleaned_words)


def verify_citations(response, retrieved_docs):
    """Replace unverified citation IDs with a security alert.

    :param response: The generated model response.
    :type response: str
    :param retrieved_docs: Documents retrieved for grounding.
    :type retrieved_docs: list
    :returns: The response with validated citations preserved.
    :rtype: str
    """
    valid_ids = {doc.metadata.get("id_chunk") for doc in retrieved_docs if doc.metadata.get("id_chunk")}

    pattern = re.compile(r"\[([^\]\s]+)\]")

    def replacer(match):
        cited_id = match.group(1)
        if cited_id in valid_ids:
            return match.group(0)
        return f"[SECURITY ALERT: Unverified citation ID - {cited_id}]"

    verified_response = pattern.sub(replacer, response)

    if "Fontes utilizadas:" in verified_response:
        citations_block = verified_response.split("Fontes utilizadas:")[1]
        found_valid_id = any(vid in citations_block for vid in valid_ids)

        if not found_valid_id:
            verified_response += "\n\n[STRUCTURAL ALERT]: The response failed to cite a valid context ID."

    return verified_response


def verify_groundedness(response, context, embeddings, threshold=0.25):
    """Check whether the response is semantically aligned with the retrieved context.

    :param response: The generated model response.
    :type response: str
    :param context: The retrieved context used for grounding.
    :type context: str
    :param embeddings: Embedding model used to compute similarity.
    :type embeddings: object
    :param threshold: Minimum similarity required for grounding.
    :type threshold: float
    :returns: The original response or a groundedness alert.
    :rtype: str
    """
    if "Não possuo informações suficientes" in response:
        return response

    response_vector = embeddings.embed_query(response)
    context_vector = embeddings.embed_query(context)

    similarity = np.dot(response_vector, context_vector) / (
        np.linalg.norm(response_vector) * np.linalg.norm(context_vector)
    )

    if similarity < threshold:
        return "[SECURITY ALERT: Vector Groundedness Check failed. Response deviates from the retrieved context.]"

    return response