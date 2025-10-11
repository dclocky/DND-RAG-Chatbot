import gradio as gr
from .rag_pipeline import answer_query

def build_app(retriever, tfidf_vectorizer, tfidf_matrix, chunks):
    return gr.Interface(
        fn=lambda q: answer_query(q, retriever, tfidf_vectorizer, tfidf_matrix, chunks),
        inputs=gr.Textbox(label="Ask a D&D Question"),
        outputs=gr.Markdown(label="Answer"),
        title="D&D RAG Assistant",
        description="Hybrid retrieval over your D&D PDFs."
    )

