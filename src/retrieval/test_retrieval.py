"""
Test Retrieval Logic
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.retrieval.search import query_financial_docs

def test_retrieval():
    print("🔬 Testing Retrieval with Metadata Filtering...")
    
    # Test 1: Broad Query
    print("\n--- Test 1: Broad Query 'autonomous driving' ---")
    results = query_financial_docs("autonomous driving", top_k=3)
    for r in results:
        print(f"   📄 [{r.get('section', '?')}] {r['child_text'][:100]}...")

    # Test 2: Strict Section Filtering (Item 1A)
    print("\n--- Test 2: 'risk factors' in Item 1A (Strict) ---")
    results = query_financial_docs(
        "competition", 
        top_k=3,
        metadata_filter={"section": "Item 1A"},
        strict_filter=True
    )
    
    if not results:
        print("   ❌ No results found (Unexpected if index is populated)")
    
    for r in results:
        print(f"   ✅ Found in [{r.get('section', '?')}]: {r['child_text'][:100]}...")
        if r.get('section') != "Item 1A":
            print(f"      Wait, got {r.get('section')}!")

    # Test 3: Cross-Filing (Item 7)
    print("\n--- Test 3: 'financial condition' in Item 7 (Strict) ---")
    results = query_financial_docs(
        "margins", 
        top_k=3,
        metadata_filter={"section": "Item 7"},
        strict_filter=True
    )
    for r in results:
        print(f"   ✅ Found in [{r.get('section', '?')}]: {r['child_text'][:100]}...")

    # Test 4: Collection Filtering
    # Note: Adjust collection_id if filename was different
    print("\n--- Test 4: Collection ID check (Strict) ---")
    results = query_financial_docs(
        "revenue", 
        top_k=3,
        metadata_filter={"collection_id": "TSLA_2023_10K"}, # Expected from pipeline
        strict_filter=True
    )
    for r in results:
        collection = r.get('collection_id', '?')
        print(f"   ✅ Found in [{collection} - {r.get('section', '?')}]: {r['child_text'][:100]}...")

if __name__ == "__main__":
    test_retrieval()
