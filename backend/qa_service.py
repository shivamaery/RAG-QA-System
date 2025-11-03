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
    """You are a scholarly research assistant.
Use only the provided context to answer the question accurately and concisely (max 150 tokens).

Context chunk:
{context}

Question:
{question}

Instructions:
- If the context contains relevant information, summarize it clearly.
- Include any explicit references to papers, authors, or sections if mentioned.
- If the context does not contain relevant information, respond exactly: "No relevant info".

Answer:"""
)

# --- Reduce Prompt ---
reduce_prompt = PromptTemplate.from_template(
    """You are a scholarly research assistant. You have been provided with partial answers from different context chunks:
{partial_answers}

Question:
{question}

Instructions:
- Combine the partial answers into a single, coherent, and concise answer.
- Cite the filenames of the source documents where information was found.
- Do not include information not present in the provided context.
- If none of the partial answers contain relevant information, respond exactly: "No relevant info".

Final Answer:"""
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
         search_type="mmr",
        search_kwargs={
            "k": k, "lambda_mult": 0.5
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
