# qa_service.py
import logging
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import chromadb
from langchain_chroma import Chroma
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain_community.embeddings import SentenceTransformerEmbeddings
import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Prompt for stuff chain
QNA_PROMPT = """
You are a scholarly, precise, and cautious research assistant. Use a formal academic tone.

Answer the question below using ONLY the provided text. 
Do NOT use outside knowledge. If the text does not contain enough information, respond exactly:
"I don't know."

Cite facts using the filename in parentheses immediately after each fact (e.g. (paper.pdf)).

Question:
{question}

Text:
{context}
"""


def get_vector_store():
    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    embeddings = SentenceTransformerEmbeddings(model_name=config.EMBEDDING_MODEL)
    return Chroma(
        client=client,
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings
    )

def build_retrieval_qa(llm, k: int = None):
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k or config.NUM_RETRIEVE,
            "fetch_k": 10 * (k or config.NUM_RETRIEVE),
            "lambda_mult": 0.5
        }
    )
    
    prompt = PromptTemplate(
        template=QNA_PROMPT,
        input_variables=["context", "question"]
    )


    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt" : prompt
        }
    )
    logger.info("Initialized RetrievalQA chain successfully.")
    return qa_chain

