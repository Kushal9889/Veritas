"""
Verify ingestion status - check checkpoint and Neo4j database state
"""
import os
import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.config import settings
from src.ingestion.storage import get_neo4j_driver, _load_checkpoint

def verify_ingestion():
    """Verify ingestion checkpoint and database state"""
    print("=" * 60)
    print("🔍 INGESTION STATUS VERIFICATION")
    print("=" * 60)
    
    checkpoint_path = os.getenv("GRAPH_CHECKPOINT", "data/ingestion/graph_checkpoint.json")
    
    # Check checkpoint file
    print(f"\n📋 Checkpoint File: {checkpoint_path}")
    checkpoint = _load_checkpoint(checkpoint_path)
    
    if checkpoint:
        last_idx = checkpoint.get("last_processed_index", -1)
        total = checkpoint.get("total_chunks", 0)
        print(f"   ✅ Last processed index: {last_idx}")
        print(f"   📊 Total chunks: {total}")
        print(f"   📈 Progress: {last_idx + 1}/{total} ({round((last_idx + 1) / total * 100, 2)}%)" if total > 0 else "   ⚠️  No progress data")
    else:
        print("   ⚠️  No checkpoint found - ingestion hasn't started yet")
    
    # Check Neo4j database state
    print(f"\n🗄️  Neo4j Database Status")
    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            # Count nodes
            result = session.run("MATCH (n) RETURN count(n) as count")
            node_count = result.single()["count"]
            
            # Count relationships
            result = session.run("MATCH ()-[r]-() RETURN count(r) as count")
            rel_count = result.single()["count"]
            
            # Get entity types
            result = session.run("MATCH (n) RETURN distinct labels(n) as types ORDER BY types")
            entity_types = [record["types"] for record in result]
            
            print(f"   ✅ Connected to Neo4j")
            print(f"   📦 Nodes in database: {node_count}")
            print(f"   🔗 Relationships in database: {rel_count}")
            print(f"   🏷️  Entity types: {entity_types if entity_types else 'None'}")
            
        driver.close()
    except Exception as e:
        print(f"   ❌ Failed to connect to Neo4j: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    if checkpoint and checkpoint.get("last_processed_index", -1) >= 0:
        last_idx = checkpoint.get("last_processed_index", -1)
        total = checkpoint.get("total_chunks", 0)
        if last_idx + 1 >= total:
            print("✅ INGESTION COMPLETE - All chunks processed!")
        else:
            print(f"⚠️  INGESTION IN PROGRESS - Ready to resume from chunk {last_idx + 2}/{total}")
    else:
        print("ℹ️  INGESTION READY TO START")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    verify_ingestion()
