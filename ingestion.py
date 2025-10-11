import os
from langchain.document_loaders import PyPDFLoader,UnstructuredWordDocumentLoader,UnstructuredExcelLoader,UnstructuredHTMLLoader,UnstructuredMarkdownLoader,UnstructuredPowerPointLoader,UnstructuredCSVLoader,UnstructuredPDFLoader,UnstructuredTextLoader,UnstructuredXMLLoader
from .config import logger

def ensure_data_path(path: str) -> None:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"DATA_PATH not found: {path}\n→ Create the folder and add PDFs.")

def load_documents(path: str):
    ensure_data_path(path)
    documents = []

    for file_name in os.listdir(path):
        file_path = Path(path) / file_name
        ext = file_path.suffix.lower()

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif ext in [".doc", ".docx"]:
                loader = UnstructuredWordDocumentLoader(str(file_path))
            elif ext in [".md", ".markdown"]:
                loader = UnstructuredMarkdownLoader(str(file_path))
            elif ext == ".csv":
                loader = CSVLoader(str(file_path))
            elif ext in [".xls", ".xlsx"]:
                loader = UnstructuredExcelLoader(str(file_path))
            else:
                logger.warning(f"Unsupported file type: {file_name}")
                continue

            docs = loader.load()
            documents.extend(docs)
            logger.info(f"Loaded {file_name} ({len(docs)} documents)")
        except Exception as e:
            logger.warning(f"Failed to load {file_name}: {e}")

    if not documents:
        raise FileNotFoundError("No supported documents found.")
    return documents
