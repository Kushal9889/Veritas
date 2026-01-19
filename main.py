import sys
from src.ingestion.vector_db import ingest_documents
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.orchestration.graph import build_graph

def run_ingestion():
    """
    runs the data pipeline: pdf -> chunks -> vector db
    """
    print("--- STARTING VERITAS INGESTION PIPELINE ---")
    
    # 1. load pdf
    pdf_path = "data/raw/sample_10k.pdf"
    print(f"📄 Loading PDF from: {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()
    print(f"✅ Loaded {len(raw_docs)} raw pages.")
    
    # 2. split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(raw_docs)
    print(f"✂️  Split into {len(chunks)} semantic chunks.")
    
    # 3. embed & index
    ingest_documents(chunks)
    print("--- PIPELINE FINISHED ---")

def run_agent(question: str):
    """
    runs the intelligent agent workflow
    """
    print(f"\n--- 🤖 VERITAS AGENT ACTIVE ---")
    print(f"❓ User Question: {question}")
    
    # 1. build the brain
    app = build_graph()
    
    # 2. run the workflow
    # inputs: the initial state with just the question
    inputs = {"question": question}
    
    # invoke the app
    result = app.invoke(inputs)
    
    # 3. print result
    print("\n--- ✅ FINAL REPORT ---")
    print(result["generation"])
    print("-----------------------")

if __name__ == "__main__":
    # simple cli interface
    # if user types 'ingest', we load data. otherwise we ask a question.
    
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        run_ingestion()
    else:
        # default mode: run the agent
        # you can change this question to test different things
        TEST_QUESTION = "how do i increase the company's revenue?"
        run_agent(TEST_QUESTION)