import sys
import os

# --- ELITE PATH INJECTION ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# ----------------------------

import google.generativeai as genai
from pinecone import Pinecone
from src.core.config import settings

def query_financial_docs(query_text: str, top_k: int = 5):
    """
    Retrieves semantic matches + metadata (page numbers, filenames).
    """
    # 1. Setup
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(settings.PINECONE_INDEX_NAME)
    genai.configure(api_key=settings.GOOGLE_API_KEY)

    print(f"🔍 Question: '{query_text}'")
    
    # 2. Embed
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=query_text,
            task_type="retrieval_query" 
        )
        query_vector = result['embedding']
    except Exception as e:
        print(f"❌ Embedding Failed: {e}")
        return []

    # 3. Query Pinecone
    print(f"📡 Searching Vector Database for top {top_k} matches...")
    try:
        search_results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
    except Exception as e:
        print(f"❌ Pinecone Search Failed: {e}")
        return []

    # 4. Format Results (NOW WITH METADATA)
    matches = []
    for match in search_results['matches']:
        metadata = match['metadata']
        
        # extracting clean info
        text_content = metadata.get('text', 'No text found.')
        source_file = metadata.get('source', 'Unknown File').split("/")[-1] # Clean the path
        # pinecone saves pages as floats sometimes, so we convert to int
        page_num = int(float(metadata.get('page', 0))) + 1 
        
        matches.append({
            "text": text_content,
            "score": match['score'],
            "source": source_file,
            "page": page_num
        })
    
    return matches

if __name__ == "__main__":
    # SMOKE TEST
    results = query_financial_docs("risk factors")
    for r in results:
        print(f"📄 {r['source']} (Page {r['page']})")