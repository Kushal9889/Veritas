from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingestion.graph_db import ingest_into_graph

def main():
    # loading the pdf again
    # just doing a subset of pages to test the graph build quickly
    print("--- 📄 loading pdf... ---")
    loader = PyPDFLoader("data/raw/sample_10k.pdf")
    pages = loader.load()
    
    # lets just take the first 5 pages for the graph test
    subset = pages[:5]
    
    print(f"--- ✂️ splitting {len(subset)} pages... ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(subset)
    
    # run the graph ingestion
    ingest_into_graph(chunks)

if __name__ == "__main__":
    main()