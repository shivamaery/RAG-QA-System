# qa_service.py
import logging
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain_community.embeddings import SentenceTransformerEmbeddings
import chromadb
import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Map Prompt ---
map_prompt = PromptTemplate.from_template(
    """System:    
You are a technical research assistant.
Use only the information in the provided context to answer the question.
Never use outside knowledge or assumptions.
Be concise and factual.

User:
Context:
{context}

Question:
{question}

Instructions:
1. If the context directly or partially answers the question, summarize the relevant details clearly (max 150 tokens).
2. If it only provides background or context, summarize how it relates.
3. If it is unrelated, respond exactly: NO RELEVANT INFO.

Assistant:
Summary: <text>
"""
)

# --- Reduce Prompt ---
reduce_prompt = PromptTemplate.from_template(
    """System:
You are a technical research assistant combining multiple partial answers.
Use only the provided summaries and their sources.
Never add information or citations that are not explicitly provided.
Produce one concise, accurate, and well-attributed answer.

User:
Partial answers (each may include its source document name or author):
{partial_answers}

Question:
{question}

Instructions:
1. Ignore any entries that read exactly “NO RELEVANT INFO.”
2. If none remain, respond exactly: NO RELEVANT INFO.
3. Otherwise:
   - Merge the remaining summaries into a single coherent answer (max 150 tokens).
   - Attribute each distinct piece of information to its source document or author if mentioned.
   - If multiple summaries agree on the same point, you may list multiple sources together.
4. Do not fabricate sources or cite documents not listed in the partial answers.

Assistant:
Final Answer: <text>"""
)


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
        document_variable_name="context"
    )

    logger.info("Initialized Map-Reduce RetrievalQA chain successfully.")
    return MapReduceQAWrapper(retriever, map_reduce_chain)
