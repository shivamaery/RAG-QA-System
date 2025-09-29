# qa_service.py
import logging
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

QNA_PROMPT = """You have been provided with the context and a question. Answer using only the context. If the answer is not in the context, respond "I don't know." Context: {context}\n\nQuestion: {question}"""

def get_vector_store():
    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    vector_store = Chroma(client=client, collection_name=config.CHROMA_COLLECTION_NAME, embedding_function=embeddings)
    return vector_store

def build_retrieval_qa(llm, k: int = None):
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={
        "k": k or config.NUM_RETRIEVE,
        "fetch_k": 10 * (k or config.NUM_RETRIEVE),  # fetch more then NUM_RETRIEVE documents
        "lambda_mult": 0.0
    })
    prompt = PromptTemplate(template=QNA_PROMPT, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs={"prompt": prompt})
    return qa_chain
