from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_pymupdf4llm import PyMuPDF4LLMParser
import chromadb

# Set quantization config
quantization_config = BitsAndBytesConfig(load_in_4bit=True, llm_int8_enable_fp32_cpu_offload=True)
# Check if GPU is available otherwise we run it on the cpu
# My initial test did not include this so I got to 
# 85% CPU usage and 17GB of RAM approximately being used
# With CUDA toolkit installed and cuda computing enabled
# on an RTX 4070Ti RAM usage was in the ballpark of 8GB while 
# GPU memory was almost saturated at 11.6/12GB
#device = "cuda" if torch.cuda.is_available() else "cpu"
# Load PDF (will need to find a better way. Something is not working with the pdf parsing, 
# which in tur makes the context window very small)

# Make sure to put the appropriate path
dir_path = "testdata2"

loader = GenericLoader(
    blob_loader=FileSystemBlobLoader(
        path=dir_path,
        glob="*.pdf",
    ),
    blob_parser=PyMuPDF4LLMParser(mode="single"),
    
)
docs = loader.load()
num_docs = len(docs)
ids = [f"id{i}" for i in range(0, num_docs)]
print(ids)
print(num_docs)


# for doc in docs:
#     print(doc.metadata)

# Create chroma db client

chroma_client = chromadb.PersistentClient(path="./test_db")
collection = chroma_client.get_or_create_collection(name="DAL_THESES")

collection.add(
    documents=docs,
    ids=ids
)


