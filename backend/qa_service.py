# qa_service.py
import logging
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain_community.embeddings import SentenceTransformerEmbeddings
import chromadb
import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Map Prompt ---
map_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a technical research assistant. "
     "You must use **only** the information in the provided context to answer the question. "
     "You **cannot** use outside knowledge, assumptions, or hallucinate facts. "
     "If the context does not provide any relevant information, respond exactly: NO RELEVANT INFO."),
    ("user",
     "Context:\n{doc_text}\n\nQuestion:\n{question}\n\n"
     "Instructions:\n"
     "1. If the context directly answers the question, summarize the answer clearly (max 150 tokens).\n"
     "2. If the context does not answer the question but provides background or related information, "
     "summarize how the context relates to the question (max 150 tokens).\n"
     "3. If the context is unrelated, respond exactly: NO RELEVANT INFO.\n\nAssistant:\nSummary:")
])

# --- Reduce Prompt ---
reduce_prompt = ChatPromptTemplate.from_messages([
     ("system",
     "You are a precise technical summarizer. Use only the provided summaries."),
    ("user",
     "Partial answers (each with its source):\n{partial_answers}\n\n"
     "Question:\n{question}\n\n"
     "Instructions:\n"
     "1. Discard any entries that are 'NO RELEVANT INFO.'\n"
     "2. Choose the most relevant and complete answer that directly addresses the question.\n"
     "3. If multiple are similar, merge minimally while preserving factual clarity.\n"
     "4. If none directly answer, respond exactly: NO RELEVANT INFO.\n\n"
     "Final Answer:")
])


# --- Vector store ---
def get_vector_store():
    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    embeddings = SentenceTransformerEmbeddings(model_name=config.EMBEDDING_MODEL)
    return Chroma(
        client=client,
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings
    )

# --- Wrapper class ---
class MapReduceQAWrapper:
    def __init__(self, retriever, map_reduce_chain):
        self.retriever = retriever
        self.chain = map_reduce_chain

    def __call__(self, query: str):
        docs = self.retriever.get_relevant_documents(query)
        logger.info("Retrieved %d documents for query.", len(docs))

        answer = self.chain.invoke({
            "input_documents": docs,
            "question": query
        })

        return {
            "result": answer,
            "answer": answer,
            "source_documents": docs
        }

# --- Build the retrieval QA ---
def build_retrieval_qa(llm, k: int = None):
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
         search_type="similarity",
        search_kwargs={
            "k": k
        }
    )

    # Map-Reduce chains
    map_chain = LLMChain(llm=llm, prompt=map_prompt)
    reduce_llm_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_llm_chain,
        document_variable_name="partial_answers"
    )

    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        token_max=2000
    )

    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="doc_text"
    )

    logger.info("Initialized Map-Reduce RetrievalQA chain successfully.")
    return MapReduceQAWrapper(retriever, map_reduce_chain)
