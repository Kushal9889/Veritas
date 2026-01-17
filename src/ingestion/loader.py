from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_pdf(file_path: str):
    """
    Ingests a PDF and splits it into semantic chunks.
    
    Args:
        file_path (str): The relative path to the PDF (e.g., 'data/raw/tesla_10k.pdf')
        
    Returns:
        List[Document]: A list of chunked documents ready for embedding.
    """
    print(f"📄 Loading PDF from: {file_path}...")
    
    # 1. EXTRACT (The "E" in ETL)
    # PyPDFLoader reads the raw bytes and converts them to a stream of text.
    loader = PyPDFLoader(file_path)
    raw_docs = loader.load()
    print(f"✅ Loaded {len(raw_docs)} raw pages.")

    # 2. TRANSFORM (The "T" in ETL)
    # We define the "Chunking Strategy". 
    # chunk_size=1000: Roughly 200-250 words. A good paragraph size.
    # chunk_overlap=100: Ensures the end of one chunk links to the start of the next.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""] # Try to split by paragraph first!
    )

    # This actually cuts the text.
    split_docs = text_splitter.split_documents(raw_docs)
    
    print(f"✂️  Split into {len(split_docs)} semantic chunks.")
    
    # Debug: Show the first chunk to ensure it looks right
    if split_docs:
        print("\n--- PREVIEW OF CHUNK 1 ---")
        print(split_docs[0].page_content[:200] + "...")
        print("--------------------------\n")
        
    return split_docs

# Simple test block to run this file directly
if __name__ == "__main__":
    # Ensure you have a PDF in data/raw/ before running this!
    # If not, create a dummy text file or download a 10-K.
    try:
        # Replace this with a real file name you drop in data/raw/
        test_path = "data/raw/sample_10k.pdf" 
        load_and_split_pdf(test_path)
    except Exception as e:
        print(f"⚠️ Error during test: {e}")