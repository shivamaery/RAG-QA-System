# cli.py
import argparse
import logging
from pathlib import Path
from model import load_phi3_model
from qa_service import build_retrieval_qa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from datetime import datetime
from pathlib import Path

def save_result_to_markdown(query: str, answer: str, sources: list):
    """
    Save the Q&A result to a local Markdown file for easy review.
    Each run is appended as a new section with timestamp and source info.
    """
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "qa_results.md"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md_content = [
        f"## {timestamp}",
        f"**Query:** {query}",
        "",
        f"**Answer:**",
        f"{answer or '*(no answer returned)*'}",
        "",
        "**Sources:**",
    ]

    if sources:
        for doc in sources:
            src = Path(doc.metadata.get("source", "unknown")).name
            md_content.append(f"- `{src}`")
    else:
        md_content.append("- *(no sources found)*")

    md_content.append("\n---\n")

    with output_file.open("a", encoding="utf-8") as f:
        f.write("\n".join(md_content))

    print(f"\nSaved to {output_file.absolute()}")

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
        qa = build_retrieval_qa(llm)
        result = qa(args.query)
        print("=== Result ===")
        print(result)
        answer = result.get("result") or result.get("answer")
        print("=== Answer ===")
        print(answer)
        sources = result.get("source_documents", [])
        print("\n=== Sources ===")
        for doc in result.get("source_documents", []):
             source_path = doc.metadata.get("source", "unknown")
             pdf_name = Path(source_path).name  # show only the file name
             print(f"- {pdf_name}")

        save_result_to_markdown(args.query, answer, sources)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
