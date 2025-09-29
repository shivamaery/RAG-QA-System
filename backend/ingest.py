# ingest.py
import logging
from pathlib import Path
from typing import List

from langchain.text_splitter import MarkdownTextSplitter
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
import uuid

import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def list_pdfs(path: Path) -> List[Path]:
    return sorted([p for p in path.glob("**/*.pdf")])


def load_documents(pdf_paths: List[Path]):
    docs = []
    for p in pdf_paths:
        loader = PyMuPDF4LLMLoader(str(p), mode="single", table_strategy="lines_strict")
        docs.extend(loader.load())
    return docs


def split_documents(docs):
    splitter = MarkdownTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    return splitter.split_documents(docs)


def create_or_get_chroma_collection(client: chromadb.Client, name: str):
    # If already exists, get it. We also check if it already has documents.
    col = client.get_or_create_collection(name=name)
    return col


def ingest_all():
    pdfs = list_pdfs(Path(config.DATA_DIR))
    if not pdfs:
        logger.warning("No PDFs found in %s", config.DATA_DIR)
        return

    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    collection = create_or_get_chroma_collection(client, config.CHROMA_COLLECTION_NAME)

    # If collection already has items, skip ingest
    if collection.count() > 0:
        logger.info("Collection '%s' already has %d items. Skipping ingest.", config.CHROMA_COLLECTION_NAME, collection.count())
        return

    logger.info("Loading %d PDFs...", len(pdfs))
    docs = load_documents(pdfs)
    logger.info("Splitting into chunks...")
    chunks = split_documents(docs)

    logger.info("Creating embeddings and adding documents to Chroma...")
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    vector_store = Chroma(client=client, collection_name=config.CHROMA_COLLECTION_NAME, embedding_function=embeddings)
    ids = [str(uuid.uuid4()) for _ in chunks]
    vector_store.add_documents(documents=chunks, ids=ids)
    # ensure persist handled by chroma client
    logger.info("Ingest complete. Added %d chunks.", len(chunks))


if __name__ == "__main__":
    ingest_all()
