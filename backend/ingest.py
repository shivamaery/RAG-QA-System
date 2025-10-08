# ingest.py
import logging
from pathlib import Path
from typing import List
import re
import uuid

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb

import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



# WIP
def clean_text(text: str) -> str:
    """
    Clean the text extracted from PDFs:
    - Remove HTML tags
    - Normalize whitespace
    - Remove common page numbers
    """
    text = re.sub(r"<.*?>", "", text) 
    text = re.sub(r"\s+", " ", text)   
    text = re.sub(r"Page \d+ of \d+", "", text, flags=re.IGNORECASE)  
    return text.strip()

def list_pdfs(path: Path) -> List[Path]:
    return sorted([p for p in path.glob("**/*.pdf")])

# HERe
def load_documents(pdf_paths: List[Path]):
    """
    Load PDFs and clean text.
    Add PDF filename as metadata for citation in QA.
    """
    docs = []
    for p in pdf_paths:
        loader = PyMuPDF4LLMLoader(str(p), mode="single", table_strategy="lines_strict")
        raw_docs = loader.load()
        for doc in raw_docs:
            doc.page_content = clean_text(doc.page_content)
            # Add source PDF filename for citation
            doc.metadata["source"] = p.name
            docs.append(doc)
    return docs


def split_documents(docs):
    """
    Split documents into semantic chunks with overlap.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?"]
    )
    return splitter.split_documents(docs)


def create_or_get_chroma_collection(client: chromadb.Client, name: str):
    # If already exists, get it.
    return client.get_or_create_collection(name=name)

def ingest_all():
    pdfs = list_pdfs(Path(config.DATA_DIR))
    if not pdfs:
        logger.warning("No PDFs found in %s", config.DATA_DIR)
        return

    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    collection = create_or_get_chroma_collection(client, config.CHROMA_COLLECTION_NAME)

    # TODO ->add things to collection
    if collection.count() > 0:
        logger.info("Collection '%s' already has %d items. Skipping ingest.", 
                    config.CHROMA_COLLECTION_NAME, collection.count())
        return

    logger.info("Loading %d PDFs...", len(pdfs))
    docs = load_documents(pdfs)

    logger.info("Splitting into chunks...")
    chunks = split_documents(docs)

    logger.info("Creating embeddings and adding documents to Chroma...")
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    vector_store = Chroma(
        client=client,
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings
    )

    # Use UUIDs as unique IDs for each chunk
    ids = [str(uuid.uuid4()) for _ in chunks]
    vector_store.add_documents(documents=chunks, ids=ids)

    logger.info("Ingest complete. Added %d chunks.", len(chunks))


if __name__ == "__main__":
    ingest_all()
