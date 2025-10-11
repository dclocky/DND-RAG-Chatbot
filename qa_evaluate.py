from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np
from .rag_pipeline import answer_query
from .config import logger

def evaluate_qa(
    retriever,
    tfidf_vectorizer,
    tfidf_matrix,
    chunks,
    qa_pairs,
    top_k: int = 5
):
    """
    Evaluate the quality of generated answers using BLEU and ROUGE-L.
    qa_pairs: List of (query, expected_answer)
    """
    bleu_scores, rouge_scores = [], []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smooth_fn = SmoothingFunction().method1

    for query, gold_answer in qa_pairs:
        generated = answer_query(query, retriever, tfidf_vectorizer, tfidf_matrix, chunks)
        bleu = sentence_bleu(
            [gold_answer.split()],
            generated.split(),
            smoothing_function=smooth_fn
        )
        rouge = scorer.score(gold_answer, generated)["rougeL"].fmeasure

        bleu_scores.append(bleu)
        rouge_scores.append(rouge)

        logger.info(f"Q: {query}")
        logger.info(f"  - BLEU:  {bleu:.3f}")
        logger.info(f"  - ROUGE: {rouge:.3f}")

    metrics = {
        "avg_bleu": np.mean(bleu_scores),
        "avg_rougeL": np.mean(rouge_scores),
    }

    logger.info("QA Evaluation Report")
    logger.info(f" - Avg BLEU:    {metrics['avg_bleu']:.3f}")
    logger.info(f" - Avg ROUGE-L: {metrics['avg_rougeL']:.3f}")
    return metrics
