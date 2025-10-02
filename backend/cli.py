# cli.py
import argparse
import logging
from pathlib import Path
from model import load_phi3_model
from qa_service import build_retrieval_qa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action="store_true", help="Run ingestion")
    parser.add_argument("--query", "-q", type=str, help="Ask a question to the system")
    args = parser.parse_args()

    if args.ingest:
        import ingest
        ingest.ingest_all()
        return

    if args.query:
        _, _, llm = load_phi3_model()
        k = 4
        qa = build_retrieval_qa(llm, k)
        result = qa(args.query)
        print("=== Result ===")
        print(result)
        answer = result.get("result") or result.get("answer")
        print("=== Answer ===")
        print(answer)
        print("\n=== Sources ===")
        for doc in result.get("source_documents", []):
             source_path = doc.metadata.get("source", "unknown")
             pdf_name = Path(source_path).name  # show only the file name
             print(f"- {pdf_name}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
