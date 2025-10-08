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

# Prompt for each chunk
QNA_CHUNK_PROMPT = """
You are a scholarly, precise, and cautious assistant. Your tone must be formal and academic.

Extract information from the text below that is directly relevant to potential questions. 
Do NOT use external knowledge or invent facts. If the text does not contain enough information to answer, respond exactly:
"I don't know."

When referencing facts, include the source PDF filename in parentheses after the relevant statement. 
Ignore any irrelevant text, HTML, or artifacts.

Text:
{text}

Source: {source}
"""

# Prompt to combine all chunk summaries
QNA_FINAL_PROMPT = """
You are a scholarly, precise, and cautious assistant. Your tone must be formal and academic.

Combine the following summarized chunks to answer the question as accurately as possible. 
Answer ONLY using the information in the summarized chunks. 
If there is not enough information, respond exactly: "I don't know."

Cite each fact with the source PDF filename in parentheses. 
Do not include any text that is not present in the summaries.

Question:
{question}

Summaries:
{summaries}
"""


def get_vector_store():
    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
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
            "lambda_mult": 0.0
        }
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
        chain_type_kwargs={"combine_prompt": final_prompt}
    )

    return qa_chain

