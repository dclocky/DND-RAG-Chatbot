from langchain.text_splitter import RecursiveCharacterTextSplitter
from .config import CHUNK_SIZE, CHUNK_OVERLAP, logger

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.page_content = chunk.page_content.replace("\n", " ").strip()
        chunk.metadata.update({
            "chunk_id": f"{chunk.metadata.get('source', 'unknown')}:{i}",
            "file_name": chunk.metadata.get("source", "unknown.pdf"),
            "chunk_index": i,
            "length": len(chunk.page_content),
        })
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks

