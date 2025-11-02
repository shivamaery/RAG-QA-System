# ingest.py
import logging
from pathlib import Path
from typing import List
import re
import uuid
import unicodedata
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
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
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"Page\s*\d+(\s*of\s*\d+)?", " ",  text, flags=re.IGNORECASE)
    text = re.sub(r"[\[\]{}*~^_]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?m)^\s*(Chapter|Section)\b.*$", "", text)
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
    )

    all_chunks = []
    for doc in docs:
        # Split and preserve metadata
        chunks = splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            chunk_doc = doc.__class__(
                page_content=chunk,
                metadata={**doc.metadata, "chunk_index": i}
            )
            all_chunks.append(chunk_doc)
    return all_chunks


def create_or_get_chroma_collection(client: chromadb.Client, name: str):
    # If already exists, get it.
    return client.get_or_create_collection(name=name)

def get_existing_sources(collection):
    """
    Return a set of all 'source' filenames already present in the collection.
    """
    existing = set()
    if collection.count() == 0:
        return existing

    results = collection.get(include=["metadatas"])
    for meta in results["metadatas"]:
        if meta and "source" in meta:
            existing.add(meta["source"])
    return existing

def add_pdfs_to_collection(client, pdf_paths):
    """
    Load, clean, split, and embed a list of PDFs into the Chroma collection.
    """
    logger.info("Loading %d PDFs...", len(pdf_paths))
    docs = load_documents(pdf_paths)

    logger.info("Splitting into chunks...")
    chunks = split_documents(docs)

    logger.info("Embedding and adding documents to Chroma...")
    embeddings = SentenceTransformerEmbeddings(model_name=config.EMBEDDING_MODEL)
    vector_store = Chroma(
        client=client,
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings
    )

    ids = [str(uuid.uuid4()) for _ in chunks]
    vector_store.add_documents(documents=chunks, ids=ids)

    logger.info("Ingest complete. Added %d chunks from %d PDFs.",
                len(chunks), len(pdf_paths))

def ingest_all():
    pdfs = list_pdfs(Path(config.DATA_DIR))
    if not pdfs:
        logger.warning("No PDFs found in %s", config.DATA_DIR)
        return

    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    collection = create_or_get_chroma_collection(client, config.CHROMA_COLLECTION_NAME)
    logger.info("Collection '%s' already has %d items", config.CHROMA_COLLECTION_NAME, collection.count())
    
    existing_sources = get_existing_sources(collection)
    new_pdfs = [p for p in pdfs if p.name not in existing_sources]

    if not new_pdfs:
        logger.info("No new PDFs to ingest.")
        return

    logger.info("Found %d new PDFs: %s", len(new_pdfs), [p.name for p in new_pdfs])
    add_pdfs_to_collection(client, new_pdfs)


if __name__ == "__main__":
    ingest_all()
