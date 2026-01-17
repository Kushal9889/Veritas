import sys
from src.ingestion.loader import load_and_split_pdf
from src.ingestion.vector_db import ingest_documents

def run_ingestion_pipeline():
    # Hardcoded for now. In Sprint 4 this comes from the user upload.
    # Make sure this matches your actual file name in data/raw
    pdf_path = "data/raw/sample_10k.pdf" 
    
    print("--- STARTING VERITAS INGESTION PIPELINE ---")
    
    # 1. Extract & Chunk
    try:
        chunks = load_and_split_pdf(pdf_path)
    except FileNotFoundError:
        print(f"❌ Error: File not found at {pdf_path}")
        print("Did you put the 10-K PDF in the 'data/raw/' folder?")
        return

    # 2. Embed & Store
    if chunks:
        ingest_documents(chunks)
    
    print("--- PIPELINE FINISHED ---")

if __name__ == "__main__":
    # Check if we are running this script directly
    run_ingestion_pipeline()