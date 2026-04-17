
import math
import re
import time
from collections import Counter
from config import TOP_K
from embeddings import embed_query
from llm import generate_response, route_query
from vector_store import query_documents
from logger import get_logger

log = get_logger(__name__)


def _tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def _bm25_score(query_tokens, doc_tokens, avg_doc_len, k1=1.5, b=0.75):
    counts  = Counter(doc_tokens)
    doc_len = len(doc_tokens)
    score   = 0.0

    for word in query_tokens:
        tf = counts.get(word, 0)
        if tf == 0:
            continue
        idf   = math.log(1 + 1 / (tf + 0.5))
        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))

    return score


def rerank(question, chunks):
    q_tokens      = _tokenize(question)
    doc_token_lists = [_tokenize(c["text"]) for c in chunks]
    avg_len       = sum(len(t) for t in doc_token_lists) / max(len(doc_token_lists), 1)

    for chunk, tokens in zip(chunks, doc_token_lists):
        chunk["bm25_score"] = _bm25_score(q_tokens, tokens, avg_len)

    return sorted(chunks, key=lambda c: c["bm25_score"], reverse=True)


def _build_prompt(chunks, question):
    context = ""
    for i, chunk in enumerate(chunks, 1):
        context += f"[Source {i} | File: {chunk['source']} | Page: {chunk['page']}]\n{chunk['text']}\n\n"

    return (
        "You are a precise document analysis assistant.\n"
        "Answer the question using ONLY the context provided below.\n\n"
        "Rules:\n"
        "- Use ONLY the information in the CONTEXT section. Do not use prior knowledge.\n"
        "- If the answer is not in the context, say: 'The answer is not available in the provided documents.'\n"
        "- Keep your answer clear and concise.\n"
        "- At the end, cite the source like this: (Source: <filename>, Page <number>)\n\n"
        f"CONTEXT:\n{context}\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )


def answer_question(question, model_name=None):
    t_start = time.perf_counter()

    routed_model, routing_reason = route_query(question, model_name)
    log.info("Model: %s | %s", routed_model, routing_reason)

    query_vector = embed_query(question)
    candidates   = query_documents(query_vector)

    if not candidates:
        return {
            "answer":         "No documents found. Please ingest documents first.",
            "sources":        [],
            "routed_model":   routed_model,
            "routing_reason": routing_reason,
            "time_taken":     round(time.perf_counter() - t_start, 2),
        }

    ranked = rerank(question, candidates)

    log.info("Top chunks:")
    for i, c in enumerate(ranked, 1):
        log.info("  [%d] Page %-4s  BM25: %.4f  %s", i, c["page"], c["bm25_score"], c["text"][:80].replace("\n", " "))

    answer  = generate_response(_build_prompt(ranked, question), routed_model)
    elapsed = round(time.perf_counter() - t_start, 2)
    log.info("Done in %.2fs", elapsed)

    return {
        "answer":         answer,
        "sources":        ranked,
        "routed_model":   routed_model,
        "routing_reason": routing_reason,
        "time_taken":     elapsed,
    }
