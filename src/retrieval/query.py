import sys
import os

# --- ELITE PATH INJECTION ---
# This ensures we can import 'src' even when running this file directly.
# It adds the project root (../../) to Python's search path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# ----------------------------

import google.generativeai as genai
from pinecone import Pinecone
from src.core.config import settings

def query_financial_docs(query_text: str, top_k: int = 3):
    """
    1. Embeds the query using Gemini.
    2. Searches Pinecone for similar vectors.
    3. Returns the matching text chunks.
    """
    # 1. Setup
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(settings.PINECONE_INDEX_NAME)
    genai.configure(api_key=settings.GOOGLE_API_KEY)

    print(f"🔍 Question: '{query_text}'")
    
    # 2. Embed the Query
    try:
        # Note: 'retrieval_query' task type is optimized for questions
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

    # 4. Format Results
    matches = []
    for match in search_results['matches']:
        # Extract the text we saved in metadata during ingestion
        text_content = match['metadata'].get('text', 'No text found.')
        score = match['score']
        matches.append({"text": text_content, "score": score})
    
    return matches

if __name__ == "__main__":
    # SMOKE TEST
    print("\n--- 🧠 VERITAS RETRIEVAL ENGINE ---")
    test_q = "What are the primary risk factors facing the company?"
    results = query_financial_docs(test_q)
    
    if not results:
        print("⚠️ No results found. Did you run ingestion?")
    
    for i, r in enumerate(results):
        print(f"\n[RESULT {i+1}] (Similarity: {r['score']:.4f})")
        print(f"{r['text'][:300]}...") # Print first 300 chars
        print("...")
    print("\n-----------------------------------")