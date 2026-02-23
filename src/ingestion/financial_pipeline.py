"""
Financial Pipeline Orchestrator

This script ties together the elite components:
1. FinancialLoader (PyMuPDF + Regex Sectioning)
2. ParentChildChunker (with metadata propagation)
3. VectorDB Ingestion (Pinecone + Metadata Validation)
"""
import glob
import os
from src.ingestion.loaders import FinancialLoader
from src.ingestion.chunkers import HybridChunker
from src.ingestion.storage import ingest_documents

def run_pipeline(data_dir: str = "data/raw"):
    print(f"🚀 Starting Financial Ingestion Pipeline from {data_dir}...")
    
    # 1. Find PDFs
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    if not pdf_files:
        print("❌ No PDF files found.")
        return

    chunker = HybridChunker()
    
    for pdf_path in pdf_files:
        print(f"\n📄 Processing {pdf_path}...")
        
        # 2. Load & Section (FinancialLoader)
        loader = FinancialLoader(pdf_path)
        # We use list(loader.lazy_load()) to get all sections first
        # Because we want to apply chunking to each section individually
        section_docs = loader.load()
        
        if not section_docs:
            print("⚠️ No content loaded.")
            continue
            
        print(f"   ✅ Loaded {len(section_docs)} high-level sections.")
        
        # 3. Advanced Chunking (Parent-Child)
        # We iterate through sections and chunk them further
        all_chunks = []
        for section_doc in section_docs:
            # We use parent_child strategy to split large sections into manageable pieces
            # passing the section metadata along
            chunks, _ = chunker.chunk_document(
                section_doc.page_content, 
                section_doc.metadata, 
                strategy="parent_child"
            )
            all_chunks.extend(chunks)
            
        print(f"   ✂️ Generated {len(all_chunks)} granular chunks with hierarchy.")
        
        # 4. Ingest to Vector DB
        dict_chunks = []
        for d in all_chunks:
            chunk_dict = {"content": d.page_content}
            chunk_dict.update(d.metadata)
            dict_chunks.append(chunk_dict)
            
        ingest_documents(
            raw_data=dict_chunks, 
            text_fields=["content"], 
            recreate_collection=(pdf_path == pdf_files[0]) # Phase 1 Idempotent reset on first PDF only.
        )
        
    print("\n🎉 Pipeline Complete.")

if __name__ == "__main__":
    run_pipeline()
