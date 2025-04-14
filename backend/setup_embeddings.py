import os
from langchain.text_splitter import MarkdownTextSplitter
from langchain_chroma import Chroma
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
import chromadb
import uuid, torch
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

# Class for our configuration params (model, chunking, etc.)
class CFG:
    model_name = 'microsoft/Phi-3-mini-128k-instruct'
    chunk_size = 1500
    chunk_overlap = 150
    temperature = 0.4
    top_p = 0.90
    repetition_penalty = 1.15
    max_len = 8192
    max_new_tokens = 512

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# Define custom prompt template for Q&a model
qna_prompt_template="""<|system|>
You have been provided with the context and a question, try to find out the answer to the question only using the context information. If the answer to the question is not found within the context, return "I dont know" as the response.<|end|>
<|user|>
Context:
{context}

Question: {question}<|end|>
<|assistant|>"""

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
    text_splitter = MarkdownTextSplitter(chunk_size=1500, chunk_overlap=150)
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
def query_vector_store(query, vector_store, chunked_docs):
    # Similarity searching, not much diversity from my tests
    # results = vector_store.similarity_search_with_relevance_scores(
    # f"{query}",
    # k=4,
    # score_threshold=0.5
    # )
    # MMR testing, optimizes both similarity and diversity
    print(f"How many docs we have: {len(chunked_docs)}\n")
    results = vector_store.max_marginal_relevance_search(f"{query}",fetch_k=len(chunked_docs), k=5,lambda_mult=0)
    for res in results:
        print(f"Query Result:\n\n {res.page_content} \n [{res.metadata}]")

# Get model and tokenizer
def get_phi_3_model():
    device = "cuda: 0" if torch.cuda.is_available() else "cpu"
     # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.float16,
        bnb_4bit_use_double_quant = True,
    ) 
    phi_3_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    quantization_config=bnb_config,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    attn_implementation="eager")
    phi_3_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    # Create pipeline for model interactions
    pipe = pipeline(
        task = "text-generation",
        model = phi_3_model,
        tokenizer = phi_3_tokenizer,
        do_sample = True,
        max_new_tokens = CFG.max_new_tokens,
        temperature = CFG.temperature,
        top_p = CFG.top_p,
        repetition_penalty = CFG.repetition_penalty,

    )

    llm = HuggingFacePipeline(pipeline=pipe)

    return phi_3_model, phi_3_tokenizer, llm

# Create Q&A chain Phi-3
def create_q_a_chain(query, model, tokenizer, llm):
    # Create prompt based off template
    PROMPT = PromptTemplate(template=qna_prompt_template, input_variables=["context", "question"])

    # Define QnA Chain
    chain = create_stuff_documents_chain(llm=llm, prompt=PROMPT)

    # messages = [{"role": "user", "content": f"{message}"}]
    # inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    # outputs = model.generate(inputs, max_new_tokens=32)
    # text = tokenizer.batch_decode(outputs)[0]
    return chain

# Utility function for answer generation
def ask(question, chain, retriever):
    context = retriever.invoke(question)
    print(f"Context: {context}\n")
    answer = chain.invoke({"context": context})
    return answer

if __name__ == '__main__':
    # Read PDFs
    chunked_docs = load_chunk_pdfs()
    # Start persistent DB
    client = chromadb.PersistentClient(path="./test_db")
    # Perform embeddings of input PDFs
    collection, vector_store_from_client = embedding_chroma(client=client, chunked_docs=chunked_docs)
    # Run query
    query = input("Enter your question: ")
    query_vector_store(query=query, vector_store=vector_store_from_client, chunked_docs=chunked_docs)
    retriever = vector_store_from_client.as_retriever(search_type="mmr", search_kwargs={'lambda_mult': 0, 'k': 4, 'fetch_k' : len(chunked_docs)})
    phi_3_model, phi_3_tokenizer, llm = get_phi_3_model()
    chain = create_q_a_chain(query=query, model=phi_3_model, tokenizer=phi_3_tokenizer, llm=llm)
    answer = ask(question=query, chain=chain, retriever=retriever)
    answer = (answer.split("<|assistant|>")[-1]).strip()
    print(f"Answer{answer}\n")
