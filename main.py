from src.config import DATA_PATH, CHROMA_PATH, EMBEDDING_MODEL
from src.ingestion import load_documents
from src.chunking import split_documents
from src.retrieval import build_tfidf, create_chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from src.evaluate import evaluate_retrieval
from src.ui import build_app

def main():
    docs = load_documents(DATA_PATH)
    chunks = split_documents(docs)
    tfidf_vectorizer, tfidf_matrix = build_tfidf(chunks)

    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    db = create_chroma(chunks, embeddings, CHROMA_PATH)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
TEST_QUERIES = [
    ("What is the main plot of Curse of Strahd?", ["vampire", "Barovia", "Strahd"]),
    ("Who is Acererak?", ["lich", "Tomb of Horrors"]),
    ("What is the goal of the Tomb of Annihilation?", ["Soulmonger", "Chult"])
]

metrics = evaluate_retrieval(
    retriever=retriever,
    tfidf_vectorizer=tfidf_vectorizer,
    tfidf_matrix=tfidf_matrix,
    chunks=chunks,
    test_queries=TEST_QUERIES,
    top_k=5
)
    app = build_app(retriever, tfidf_vectorizer, tfidf_matrix, chunks)
    app.launch()

if __name__ == "__main__":
    main()
