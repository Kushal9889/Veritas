import time
import uuid
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from src.core.config import settings
from langchain_core.documents import Document

def get_pinecone_client():
    return Pinecone(api_key=settings.PINECONE_API_KEY)

def configure_gemini():
    """
    Authenticates the Google GenAI SDK.
    """
    genai.configure(api_key=settings.GOOGLE_API_KEY)

def ensure_index_exists(pc: Pinecone, index_name: str):
    """
    Checks if index exists, creates it if not.
    """
    existing_indexes = [i.name for i in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"⚠️ Index '{index_name}' not found. Creating (Serverless)...")
        # NOTE: Google embeddings are 768 dimensions (usually), but the newer 
        # text-embedding-004 can also do higher. 
        # Standard text-embedding-004 output dimension is 768.
        # We MUST ensure Pinecone matches this.
        pc.create_index(
            name=index_name,
            dimension=768, # <--- CRITICAL CHANGE: Google uses 768, OpenAI used 1536
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        print("✅ Index ready.")

def ingest_documents(docs: list[Document]):
    if not docs:
        return

    print(f"🔌 Connecting to Pinecone Index: {settings.PINECONE_INDEX_NAME}")
    
    # 1. Initialize
    pc = get_pinecone_client()
    configure_gemini()
    
    # CRITICAL: We need to handle the dimension mismatch.
    # If you created the index for OpenAI (1536 dims), Google (768 dims) will fail.
    # We will try to delete and recreate if the dimensions are wrong, 
    # but for now, assuming a fresh start is safer.
    ensure_index_exists(pc, settings.PINECONE_INDEX_NAME)
    index = pc.Index(settings.PINECONE_INDEX_NAME)

    batch_size = 50 
    print(f"🚀 Starting ingestion of {len(docs)} chunks using Gemini...")

    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        texts = [d.page_content.replace("\n", " ") for d in batch]
        metadatas = [d.metadata for d in batch]
        
        print(f"   🔹 Embedding batch {i//batch_size + 1}...")
        try:
            # GOOGLE EMBEDDING CALL
            # Model: models/text-embedding-004
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=texts,
                task_type="retrieval_document"
            )
            vectors = result['embedding']
        except Exception as e:
            print(f"❌ Gemini Embedding Error: {e}")
            continue

        to_upsert = []
        for text, vector, meta in zip(texts, vectors, metadatas):
            meta["text"] = text 
            to_upsert.append((str(uuid.uuid4()), vector, meta))

        try:
            index.upsert(vectors=to_upsert)
            print(f"   ✅ Batch {i//batch_size + 1} uploaded.")
        except Exception as e:
            print(f"   ❌ Pinecone Upsert Error: {e}")

    print("🎉 Ingestion Pipeline Complete.")