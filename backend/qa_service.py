# qa_service.py
import logging
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
import chromadb
from langchain_chroma import Chroma
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.embeddings import SentenceTransformerEmbeddings
import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Prompt for stuff chain
QNA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a scholarly research assistant. Use a formal academic tone. "
     "Answer the user’s questions based only on the provided context, "
     "and cite the filenames of the source documents used."),
    ("user",
     "Use the following context to answer the question. "
     "Do NOT include text that is not supported by the context.\n\n"
     "Context:\n{context}\n\nQuestion:\n{question}\n\n"
     "Provide a clear, concise answer.")
])



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
    

    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": QNA_PROMPT},
        return_source_documents=True
    )
    logger.info("Initialized RetrievalQA chain successfully.")
    return qa_chain

