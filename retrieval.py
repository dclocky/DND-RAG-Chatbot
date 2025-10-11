import shutil
import numpy as np
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from .config import logger

def build_tfidf(chunks) -> Tuple[TfidfVectorizer, np.ndarray]:
    texts = [c.page_content for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix

def create_chroma(chunks, embeddings, persist_dir, rebuild=True) -> Chroma:
    if rebuild:
        shutil.rmtree(persist_dir, ignore_errors=True)
    ids = [c.metadata["chunk_id"] for c in chunks]
    db = Chroma.from_documents(chunks, embeddings, ids=ids, persist_directory=persist_dir)
    db.persist()
    logger.info(f" Chroma DB ready: {persist_dir}")
    return db

def hybrid_retriever(
    query,
    retriever,
    tfidf_vectorizer,
    tfidf_matrix,
    chunks,
    top_k: int = 5,
    alpha: float = 0.5
) -> List:
    """
    Hybrid retriever with proper score normalization.
    Combines semantic (Chroma) and lexical (TF-IDF) similarity with weight alpha.
    """

    sem_docs = retriever.get_relevant_documents(query)
    sem_scores_dict = {doc.metadata["chunk_id"]: doc.metadata.get("score", 1.0) for doc in sem_docs}

    sem_scores = np.array([sem_scores_dict.get(c.metadata["chunk_id"], 0.0) for c in chunks])
    if sem_scores.max() > 0:
        sem_scores = (sem_scores - sem_scores.min()) / (sem_scores.max() - sem_scores.min() + 1e-8)

    query_vec = tfidf_vectorizer.transform([query])
    tfidf_scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    if tfidf_scores.max() > 0:
        tfidf_scores = (tfidf_scores - tfidf_scores.min()) / (tfidf_scores.max() - tfidf_scores.min() + 1e-8)

    combined_scores = alpha * sem_scores + (1 - alpha) * tfidf_scores

    ranked_indices = np.argsort(combined_scores)[::-1]
    top_indices = ranked_indices[:top_k]
    results = [chunks[i] for i in top_indices]

    logger.info(f"Hybrid retrieval completed | α={alpha:.2f} | top_k={top_k}")
    return results
