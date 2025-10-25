# qa_service.py
import logging
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
import chromadb
from langchain_chroma import Chroma
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import SentenceTransformerEmbeddings
import config
from langchain.chains import LLMChain

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Prompt for stuff chain
# QNA_PROMPT = ChatPromptTemplate.from_messages([
#     ("system",
#      "You are a scholarly research assistant. Use a formal academic tone. "
#      "Answer the user’s questions based only on the provided context, "
#      "and cite the filenames of the source documents used."),
#     ("user",
#      "Use the following context to answer the question. "
#      "Do NOT include text that is not supported by the context.\n\n"
#      "Context:\n{context}\n\nQuestion:\n{question}\n\n"
#      "Provide a clear, concise answer.")
# ])

# Map prompt
map_prompt = PromptTemplate.from_template(
    """You are a scholarly research assistant.
    Use only the provided context to answer the question concisely (max 150 tokens).
    Context chunk:
    {context}

    Question:
    {question}

    Provide a short answer, or "No relevant info" if not applicable.
    """
)
# Reduce prompt
reduce_prompt = PromptTemplate.from_template(
    """You are a scholarly research assistant. You have been provided with partial answers from different context chunks:
    {partial_answers}
    
    Question: {question}
    
    Combine the partial answers into one coherent, clear and concise final answer.
    Cite the filenames of the source documents used.
    """
)




def get_vector_store():
    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    embeddings = SentenceTransformerEmbeddings(model_name=config.EMBEDDING_MODEL)
    return Chroma(
        client=client,
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings
    )

class MapReduceQAWrapper:
    def __init__(self, retriever, map_reduce_chain):
        self.retriever = retriever
        self.chain = map_reduce_chain

    def __call__(self, query: str):
        # Retrieve documents using the existing retriever
        
        docs = self.retriever.get_relevant_documents(query)

        # Run the Map-Reduce chain
        answer = self.chain.invoke({"input_documents": docs, "question": query})

        # Return a dictionary to match RetrievalQA
        return {
            "result": answer,
            "answer": answer,
            "source_documents": docs
        }
    
def build_retrieval_qa(llm, k: int = None):
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
        "k": k or config.NUM_RETRIEVE
    }
    )
    
    map_chain = LLMChain(llm=llm, prompt=map_prompt)
    reduce_llm_chain = LLMChain(llm=llm, prompt=reduce_prompt)
    
    combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_llm_chain,
    document_variable_name="partial_answers"
)
    reduce_documents_chain = ReduceDocumentsChain(
    combine_documents_chain=combine_documents_chain,
    token_max=3500,              
    collapse_single_docs=True
)
    map_reduce_chain = MapReduceDocumentsChain(
    llm_chain=map_chain,                  # map step
    reduce_documents_chain=reduce_documents_chain,  # reduce step
    document_variable_name="context"     
)
    
    # Create the RetrievalQA chain
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     retriever=retriever,
    #     chain_type="stuff",
    #     chain_type_kwargs={"prompt": QNA_PROMPT},
    #     return_source_documents=True
    # )
    logger.info("Initialized RetrievalQA chain successfully.")
    return MapReduceQAWrapper(retriever, map_reduce_chain)

