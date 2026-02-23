import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_community.document_loaders import PyPDFLoader
from src.ingestion.storage import ingest_into_graph

def reingest_neo4j(max_pages=3): # limit to 3 pages for testing
    print("--- STARTING NEO4J GRAPH RE-INGESTION ---")
    pdf_path = "data/raw/sample_10k.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ Could not find {pdf_path}. Make sure it exists.")
        return

    print(f"📄 Loading PDF from: {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()

    if max_pages:
        raw_docs = raw_docs[:max_pages]
        print(f"✂️  Limited to first {max_pages} pages.")

    print(f"✅ Loaded {len(raw_docs)} pages to process.")
    
    print("🔄 Chunking...")
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(raw_docs)
    
    print(f"✂️  Created {len(docs)} chunks for Neo4j extraction.")

    print("\n--- INGESTING INTO NEO4J ---")
    ingest_into_graph(docs)
    print("--- FINISHED NEO4J GRAPH INGESTION ---")

if __name__ == "__main__":
    max_test_pages = 5
    print(f"Running Neo4j re-ingestion script with max_pages={max_test_pages}")
    reingest_neo4j(max_pages=max_test_pages)
