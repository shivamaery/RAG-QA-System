# qa_service.py
import logging
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Prompt for each chunk
QNA_CHUNK_PROMPT = """
You are a scholarly, precise, and cautious assistant. Use a formal academic tone.

Task: Using ONLY the text below, extract any facts or statements that directly answer the question.
Do NOT use external knowledge or invent facts. If the text does not contain enough information to answer, respond exactly:
"I don't know."

When you list facts, append the source filename in parentheses immediately after each fact (e.g. (paper.pdf)).
Do not provide commentary, speculation, or extra wording.

Question:
{question}

Text:
{context}

Source: {source}
"""

# Prompt to combine all chunk summaries
QNA_FINAL_PROMPT = """
You are a scholarly, precise, and cautious assistant. Use a formal academic tone.

Combine the following chunk-level extracted facts to answer the question as accurately as possible.
Answer ONLY using the information in the chunk extracts. If there is not enough information, respond exactly: "I don't know."

Cite each fact with the source PDF filename in parentheses. Do not add or invent any text not present in the extracts.

Question:
{question}

Chunk-level extracted facts:
{summaries}
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
    
    map_prompt = PromptTemplate(
    template=QNA_CHUNK_PROMPT,
    input_variables=["context", "question", "source"]
    )


    final_prompt = PromptTemplate(
        template=QNA_FINAL_PROMPT,
        input_variables=["summaries", "question"]
    )

    # Only combine_prompt is supported
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
               "map_prompt": map_prompt,
               "combine_prompt": final_prompt,
               "return_intermediate_steps": True}
    )
    logger.info("Initialized RetrievalQA chain successfully.")
    return qa_chain

