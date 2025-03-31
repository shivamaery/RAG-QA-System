import os
from langchain.text_splitter import MarkdownTextSplitter
from langchain_chroma import Chroma
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
import chromadb
import uuid
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Load PDFs as markdowns and chunk them
def load_chunk_pdfs():
    pdfs_path = "test_data2"
    # Initialize empty list of documents
    docs = []
    for file in os.listdir(pdfs_path):
        if (file.endswith('.pdf')):
            pdf = os.path.join(pdfs_path, file)
            loader = PyMuPDF4LLMLoader(pdf, mode="single",table_strategy='lines_strict')
            doc = loader.load()
            docs.extend(doc)
    text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunked_docs = text_splitter.split_documents(docs)
    
    return chunked_docs
    
# Embed Pdfs into chromadb local persistent instance and create vector store
def embedding_chroma(client, chunked_docs):
    ids = []
    for i in range(len(chunked_docs)):
        id = str(uuid.uuid4())
        ids.append(id)
    collection = client.get_or_create_collection(name="DAL_THESES")
    vector_store_from_client = Chroma(client=client, collection_name="DAL_THESES", embedding_function=embeddings)
    vector_store_from_client.add_documents(documents=chunked_docs, ids=ids)
    return collection, vector_store_from_client

# Query vector store
def query_vector_store(query, vector_store):
    results = vector_store.similarity_search(
    f"{query}",
    k=3)
    for res in results:
        print(f"QUERY RESULT: \n{res.page_content} \n{res.metadata}")

if __name__ == '__main__':
    chunked_docs = load_chunk_pdfs()
    client = chromadb.PersistentClient(path="./test_db")
    collection, vector_store_from_client = embedding_chroma(client, chunked_docs)
    query_vector_store("Finite Element", vector_store_from_client)
    #chroma_query(collection)