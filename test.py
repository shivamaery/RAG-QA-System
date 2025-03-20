from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Check if GPU is available otherwise we run it on the cpu
# My initial test did not include this so I got to 
# 85% CPU usage and 17GB of RAM approximately being used
# With CUDA toolkit installed and cuda computing enabled
# on an RTX 4070Ti RAM usage was in the ballpark of 8GB while 
# GPU memory was almost saturated at 11.6/12GB
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load PDF (will need to find a better way. Something is not working with the pdf parsing, 
# which in tur makes the context window very small)

# Make sure to put the appropriate path
input_dir = r"test_data"

documents = SimpleDirectoryReader(input_dir=input_dir).load_data()

# print(documents[0].text[:15000])  # Debug, checking document content

# embedding
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# vector index
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# persist index into a json
index.storage_context.persist(persist_dir="./index_storage")

model_name = "microsoft/phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device) # Move to device depending on GPU availability

def query_model(query):
    retriever = index.as_retriever()
    context = retriever.retrieve(query)

    prompt = f"Based on the research papers, answer the following:\n\n{context}\n\nQuestion: {query}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device) # Move input tensors to device depending on GPU availability
    output = model.generate(**inputs, max_new_tokens=200)
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

question = "What are the key findings of Hamadou Saliah-Hassane's Thesis"
answer = query_model(question)
print(answer)