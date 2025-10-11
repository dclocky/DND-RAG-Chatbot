from .retrieval import hybrid_retriever
from .llm import t5_generate
from .config import truncate_to_max_tokens, MAX_TOKENS, logger

def answer_query(query, retriever, tfidf_vectorizer, tfidf_matrix, chunks):
    logger.info(f"🔍 Query: {query}")
    top_docs = hybrid_retriever(query, retriever, tfidf_vectorizer, tfidf_matrix, chunks, top_k=10)

    summaries = []
    for d in top_docs:
        snippet = truncate_to_max_tokens(d.page_content, 300)
        summary = t5_generate(
            f"Summarize D&D content clearly:\n{snippet}",
            max_new_tokens=160
        )
        summaries.append(summary.strip())

    context = truncate_to_max_tokens("\n\n".join(summaries), MAX_TOKENS)
    final_prompt = f"Use this context to answer:\n{context}\n\nQuestion: {query}\nAnswer:"
    return t5_generate(final_prompt, max_new_tokens=220)
