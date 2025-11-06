# api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

from model import load_phi4_model
from qa_service import build_retrieval_qa
import config

# Creating app object
app = FastAPI()

# Allow React frontend to call API
# Disclaimer: Since this application will only be ran locally the next
# piece of code which handles cross origins is fine to keep as is but would.
# otherwise leave the API vulnerable.
# This is essentially allowing all origins to interact with the api given
# no domains are specified and only "*" is present.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Globals
llm = None
qa_chain = None

# Load model on startup
@app.on_event("startup")
async def startup_event():
    global llm, qa_chain
    logger.info("Loading model and QA chain...")
    _, _, llm = load_phi4_model()
    qa_chain = build_retrieval_qa(llm, k=config.NUM_RETRIEVE)
    logger.info("Model and retrieval QA chain ready.")

# Request/Response models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


# Using post in order to send the query and get the answer
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    try:
        result = qa_chain(req.question)

        # Temporary solution to print the whole returned answer if not a string.
        raw_answer = result.get("result") or result.get("answer")
        if isinstance(raw_answer, dict):
            # Extract text from dict or convert to string
            answer = str(raw_answer.get("text", raw_answer))
        else:
            answer = str(raw_answer)

        sources = [str(doc.metadata.get("source", "unknown")) for doc in result.get("source_documents", [])]

        return QueryResponse(answer=answer, sources=sources)

    except Exception as e:
        logger.exception("Error while processing query: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

# Entrypoint to run the FastAPI backend
# Check if frontend isnt running on the same port
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000)