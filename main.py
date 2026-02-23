import sys
import os
import json
from src.ingestion.storage import ingest_documents, ingest_into_graph
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.orchestration.workflow import build_graph
from src.core.telemetry import get_telemetry

# Initialize telemetry
telemetry = get_telemetry('main')

def run_ingestion(max_pages=None, *, graph_only: bool = False, resume_graph: bool = False, clear_graph: bool = True, checkpoint_path: str | None = None):
    """Run data pipeline with parent-child chunking"""
    telemetry.log_info("Starting ingestion pipeline", max_pages=max_pages)
    print("--- STARTING VERITAS INGESTION PIPELINE ---")
    
    pdf_path = "data/raw/sample_10k.pdf"
    print(f"📄 Loading PDF from: {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()
    
    if max_pages:
        raw_docs = raw_docs[:max_pages]
        print(f"✂️  Limited to first {max_pages} pages.")
    
    telemetry.log_info("PDF loaded", pages=len(raw_docs))
    print(f"✅ Loaded {len(raw_docs)} raw pages.")
    
    # Create parent-child chunks
    print("🔄 Chunking...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    child_chunks = text_splitter.split_documents(raw_docs)
    
    telemetry.log_info("Chunking completed", child_chunks=len(child_chunks))
    print(f"✂️  Created {len(child_chunks)} chunks.")
    
    # Needs to match signature `ingest_documents(raw_data: List[Dict], text_fields: List[str], recreate_collection: bool = False)`
    # Wait, in Phase 3 storage.py I refactored ingest_documents to accept dictionaries.
    # Let me format them.
    dict_chunks = [{"text": doc.page_content, "page_num": doc.metadata.get("page", 0), "source": "sample_10k.pdf", "section": "Item 1"} for doc in child_chunks]

    # Ingest child chunks (for precise retrieval)
    if not graph_only:
        try:
            ingest_documents(dict_chunks, text_fields=["text"], recreate_collection=True)
        except Exception as e:
            telemetry.log_warning("Vector ingestion failed; continuing to graph ingestion", error=str(e))
            print(f"⚠️ Vector ingestion failed (continuing): {e}")
    
    print("\n--- STARTING GRAPH INGESTION ---")
    ingest_into_graph(
        child_chunks,
        clear=clear_graph and not resume_graph,
        resume_from_checkpoint=resume_graph,
        checkpoint_path=checkpoint_path or "data/ingestion/graph_checkpoint.json",
    )
    
    telemetry.log_info("Ingestion pipeline completed")
    print("--- PIPELINE FINISHED ---")

def run_agent(question: str):
    """Run intelligent agent workflow with streaming output"""
    import os
    from src.core.config import settings
    
    telemetry.log_info("Starting agent", question=question)
    print(f"\n--- 🤖 VERITAS AGENT ACTIVE ---")
    print(f"❓ User Question: {question}")
    print("\n--- ✅ RESPONSE ---\n")
    
    # X3: LangSmith setup is handled by config.py._setup_langsmith() — no override needed
    app = build_graph()
    inputs = {"question": question}
    
    result = app.invoke(inputs)
    
    response = result["generation"]
    telemetry.log_info("Agent completed", response_length=len(response))
    
    print("\n\n" + "="*60)
    print(f"📊 Response Stats: {len(response)} characters, {len(response.split())} words")
    print("="*60)

if __name__ == "__main__":
    # simple cli interface
    # Usage examples:
    # python main.py query "Your question here"  -> query with custom question
    # python main.py ingest                      -> ingest all pages
    # python main.py ingest 5                    -> ingest first 5 pages only
    # python main.py                             -> run agent with default question
    
    if len(sys.argv) > 1 and sys.argv[1] == "query":
        # Query mode with custom question
        if len(sys.argv) > 2:
            question = " ".join(sys.argv[2:])
        else:
            question = "What are the main revenue streams?"
        run_agent(question)
    elif len(sys.argv) > 1 and sys.argv[1] == "ingest":
        max_pages = None
        graph_only = False
        resume_graph = False
        clear_graph = True
        checkpoint_path = None

        args = sys.argv[2:]
        i = 0
        while i < len(args):
            arg = args[i]
            if arg.isdigit():
                max_pages = int(arg)
                print(f"🔍 Testing mode: Processing only {max_pages} pages")
            elif arg == "--graph-only":
                graph_only = True
            elif arg == "--resume-graph":
                resume_graph = True
                clear_graph = False
            elif arg == "--no-clear-graph":
                clear_graph = False
            elif arg == "--clear-graph":
                clear_graph = True
            elif arg == "--checkpoint" and i + 1 < len(args):
                checkpoint_path = args[i + 1]
                i += 1
            else:
                print(f"⚠️  Unknown ingest arg: {arg}")
            i += 1

        run_ingestion(
            max_pages,
            graph_only=graph_only,
            resume_graph=resume_graph,
            clear_graph=clear_graph,
            checkpoint_path=checkpoint_path,
        )
    else:
        # default mode: run the agent with default question
        TEST_QUESTION = "What are Tesla's main revenue streams?"
        run_agent(TEST_QUESTION)
