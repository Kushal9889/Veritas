import time
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from src.core.config import settings

def get_vectorstore():
    """
    Returns the Pinecone vector store interface.
    """
    # using small model for speed/cost. 3-large is overkill for sprint 1.
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", 
        api_key=settings.OPENAI_API_KEY
    )
    
    # Existing index check handled by Pinecone internally usually, 
    # but we need the client to manage index creation if missing.
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    
    index_name = settings.PINECONE_INDEX_NAME

    # Check if index exists. If not, create it.
    # standard tier = serverless. 
    existing_indexes = [i.name for i in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"⚠️ Index '{index_name}' not found. Creating...")
        pc.create_index(
            name=index_name,
            dimension=1536, # matches text-embedding-3-small
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        # give AWS a second to spin it up
        time.sleep(1)

    # attach to the index
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=settings.PINECONE_API_KEY
    )
    
    return vector_store

def ingest_documents(docs: list[Document]):
    """
    Batched upload to Pinecone. 
    """
    if not docs:
        print("No docs to ingest. Skipping.")
        return

    print(f"🔌 Connecting to Pinecone Index: {settings.PINECONE_INDEX_NAME}")
    vector_store = get_vectorstore()

    print(f"🚀 Uploading {len(docs)} chunks...")
    
    # LangChain handles batching under the hood, but explicitly 
    # add_documents is robust.
    # Try/catch block crucial here - network flakes happen.
    try:
        vector_store.add_documents(docs)
        print("✅ Ingestion complete.")
    except Exception as e:
        print(f"❌ Failed to upload vectors: {e}")
        # real production code would retry here
        raise e