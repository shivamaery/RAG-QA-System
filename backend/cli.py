# cli.py
import argparse
import logging
from pathlib import Path
from model import load_phi4_model
from qa_service import build_retrieval_qa
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import config
 
# Create a single console instance
console = Console()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_result(query: str, answer: str, sources: list):
    """Pretty-print the QA result using Rich panels and tables."""
    # Query panel
    console.print(Panel(f"[bold blue]{query}[/bold blue]", title="Question", expand=True))

    # Answer panel
    console.print(Panel(f"[green]{answer or '*(no answer returned)*'}[/green]", title="Answer", expand=True))

    # Sources table
    table = Table(title="Sources", show_header=True, header_style="bold yellow")
    table.add_column("#", style="yellow", width=3)
    table.add_column("PDF File", style="cyan")

    if sources:
        for idx, doc in enumerate(sources, start=1):
            pdf_name = Path(doc.metadata.get("source", "unknown")).name
            table.add_row(str(idx), pdf_name)
    else:
        table.add_row("-", "*(no sources found)*")

    console.print(table)


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
        _, _, llm = load_phi4_model()
        qa = build_retrieval_qa(llm=llm, k=config.NUM_RETRIEVE)
        result = qa(args.query)
        answer = result.get("output_text") or result.get("answer") or result.get("result")
        sources = result.get("source_documents", [])
        print_result(args.query, answer, sources)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
