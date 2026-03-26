from embeddings import embed_query
from vector_store import query_documents
from llm import generate_response


def build_prompt(context_chunks: list[str], question: str) -> str:
    """
    Construct the full prompt for the LLM.

    This is the most important function in RAG.
    The prompt tells the LLM to ONLY use the provided context.
    This prevents hallucination by grounding answers in documents.
    """
    # Join retrieved chunks into one context block
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not found in the context, say "I don't know based on the provided documents."
Do not use any prior knowledge outside the context.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    return prompt


def answer_question(
    question: str,
    model_name: str = None
) -> dict:
    """
    Full RAG query pipeline.

    Steps:
    1. Embed the question
    2. Find similar chunks in ChromaDB
    3. Build a grounded prompt
    4. Get LLM response

    Returns dict with 'answer' and 'sources'.
    """
    # Step 1: Embed the query
    query_vec = embed_query(question)

    # Step 2: Retrieve relevant chunks
    relevant_chunks = query_documents(query_vec)

    if not relevant_chunks:
        return {
            "answer": "No documents found in the knowledge base. Please ingest documents first.",
            "sources": []
        }

    # Step 3: Build the prompt
    prompt = build_prompt(relevant_chunks, question)

    # Step 4: Generate answer with local LLM
    answer = generate_response(prompt, model_name)

    return {
        "answer": answer,
        "sources": relevant_chunks   # Return for transparency
    }