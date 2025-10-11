import argparse
from langchain.embeddings import SentenceTransformerEmbeddings

from .config import DATA_PATH, CHROMA_PATH, EMBEDDING_MODEL
from .ingestion import load_documents
from .chunking import split_documents
from .retrieval import build_tfidf, create_chroma
from .evaluate import evaluate_retrieval, log_metrics_to_csv
from .qa_evaluate import evaluate_qa

TEST_QUERIES = [
    ("What is the main plot of Curse of Strahd?", ["vampire", "Barovia", "Strahd"]),
    ("Who is Acererak?", ["lich", "Tomb of Horrors"]),
    ("What is the Soulmonger?", ["soulmonger", "Chult", "death curse"])
]

QA_PAIRS = [
    ("Who is Acererak?", "Acererak is a powerful lich and the antagonist of Tomb of Horrors."),
    ("What is the main plot of Curse of Strahd?", "Players must defeat Strahd von Zarovich and free Barovia from his curse."),
]

def prepare_pipeline():
    """Build pipeline once for both retrieval and QA evaluation."""
    docs = load_documents(DATA_PATH)
    chunks = split_documents(docs)
    tfidf_vectorizer, tfidf_matrix = build_tfidf(chunks)

    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    db = create_chroma(chunks, embeddings, CHROMA_PATH)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return retriever, tfidf_vectorizer, tfidf_matrix, chunks

def run_retrieval_eval():
    retriever, tfidf_vectorizer, tfidf_matrix, chunks = prepare_pipeline()
    metrics = evaluate_retrieval(
        retriever, tfidf_vectorizer, tfidf_matrix, chunks, TEST_QUERIES, top_k=5
    )
    log_metrics_to_csv(metrics, strategy_name="hybrid_v1")

def run_qa_eval():
    retriever, tfidf_vectorizer, tfidf_matrix, chunks = prepare_pipeline()
    qa_metrics = evaluate_qa(
        retriever, tfidf_vectorizer, tfidf_matrix, chunks, QA_PAIRS
    )
    print("QA Metrics:", qa_metrics)

def main():
    parser = argparse.ArgumentParser(description="RAG Assistant CLI")
    parser.add_argument(
        "command", choices=["evaluate", "qa"], help="Run retrieval or QA evaluation"
    )
    args = parser.parse_args()

    if args.command == "evaluate":
        run_retrieval_eval()
    elif args.command == "qa":
        run_qa_eval()

if __name__ == "__main__":
    main()
