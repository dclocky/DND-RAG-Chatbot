import time
import numpy as np
from typing import List, Tuple
from sklearn.metrics import precision_score, recall_score

from .retrieval import hybrid_retriever
from .config import logger

def precision_at_k(relevant_docs: List[int], retrieved_docs: List[int], k: int) -> float:
    """Compute Precision@k."""
    retrieved_at_k = retrieved_docs[:k]
    if not retrieved_at_k:
        return 0.0
    return len(set(retrieved_at_k) & set(relevant_docs)) / len(retrieved_at_k)

def recall_at_k(relevant_docs: List[int], retrieved_docs: List[int], k: int) -> float:
    """Compute Recall@k."""
    retrieved_at_k = retrieved_docs[:k]
    if not relevant_docs:
        return 0.0
    return len(set(retrieved_at_k) & set(relevant_docs)) / len(relevant_docs)

def mean_reciprocal_rank(all_rank_positions: List[int]) -> float:
    """Compute Mean Reciprocal Rank (MRR)."""
    if not all_rank_positions:
        return 0.0
    return np.mean([1.0 / r if r > 0 else 0.0 for r in all_rank_positions])

def evaluate_retrieval(
    retriever,
    tfidf_vectorizer,
    tfidf_matrix,
    chunks,
    test_queries: List[Tuple[str, List[str]]],
    top_k: int = 5
):
    """
    Evaluate retrieval quality using precision@k, recall@k, MRR, and latency.
    test_queries: List of (query, [expected_keywords])
    """
    total_latency = 0
    precisions, recalls, ranks = [], [], []

    for query, expected_keywords in test_queries:
        start = time.time()
        docs = hybrid_retriever(query, retriever, tfidf_vectorizer, tfidf_matrix, chunks, top_k=top_k)
        elapsed = time.time() - start
        total_latency += elapsed

        retrieved_texts = [c.page_content.lower() for c in docs]

        rank_position = 0
        relevant_indices = []
        for kw in expected_keywords:
            for i, text in enumerate(retrieved_texts):
                if kw.lower() in text:
                    relevant_indices.append(i)
                    if rank_position == 0 or i + 1 < rank_position:
                        rank_position = i + 1
                    break

        ranks.append(rank_position if rank_position > 0 else 0)

        precisions.append(precision_at_k(relevant_indices, list(range(len(docs))), top_k))
        recalls.append(recall_at_k(relevant_indices, list(range(len(docs))), top_k))

    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_mrr = mean_reciprocal_rank(ranks)
    avg_latency = total_latency / len(test_queries)

    logger.info("Retrieval Evaluation Report")
    logger.info(f" - Precision@{top_k}: {avg_precision:.3f}")
    logger.info(f" - Recall@{top_k}:    {avg_recall:.3f}")
    logger.info(f" - MRR:               {avg_mrr:.3f}")
    logger.info(f" - Avg Latency:       {avg_latency:.3f}s")

    return {
        "precision@k": avg_precision,
        "recall@k": avg_recall,
        "mrr": avg_mrr,
        "avg_latency": avg_latency
    }
def log_metrics_to_csv(metrics: dict, strategy_name: str, filename: str = "eval_results.csv"):
    """
    Append metrics to a CSV file so you can compare runs over time.
    Each row: timestamp, strategy_name, precision, recall, mrr, latency
    """
    metrics_row = {
        "timestamp": datetime.utcnow().isoformat(),
        "strategy": strategy_name,
        **metrics
    }
    fieldnames = ["timestamp", "strategy", "precision@k", "recall@k", "mrr", "avg_latency"]

    write_header = not os.path.exists(filename)
    with open(filename, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(metrics_row)


    logger.info(f"Metrics logged to {filename}")
