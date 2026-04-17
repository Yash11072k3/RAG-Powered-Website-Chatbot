import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def split_text(text: str, chunk_size: int = 900, overlap: int = 150):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            if chunks and overlap > 0:
                prev = chunks[-1]
                overlap_text = prev[-overlap:] if len(prev) > overlap else prev
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk = sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def create_embeddings(chunks):
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return np.array(embeddings, dtype="float32")


def store_embeddings(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def keyword_score(query: str, chunks):
    query_words = [w.strip().lower() for w in query.split() if w.strip()]
    scores = []

    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = sum(chunk_lower.count(word) for word in query_words)
        scores.append(score)

    return scores


def hybrid_search(query: str, index, chunks, top_k: int = 5):
    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    search_k = min(max(top_k * 4, 12), len(chunks))
    similarities, indices = index.search(query_vector, search_k)
    keyword_scores = keyword_score(query, chunks)

    results = []
    for rank, idx in enumerate(indices[0]):
        semantic_score = float(similarities[0][rank])
        lexical_score = float(keyword_scores[idx])
        combined_score = semantic_score + (lexical_score * 0.08)

        results.append({
            "id": int(idx),
            "text": chunks[idx],
            "score": combined_score
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]