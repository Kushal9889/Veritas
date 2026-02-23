"""
Real-time ingestion progress monitor
"""
import os
import sys
import json
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.config import settings
from src.ingestion.storage import get_neo4j_driver, _load_checkpoint

def monitor_progress(interval=5, max_iterations=None):
    """Monitor ingestion progress in real-time"""
    print("=" * 70)
    print("📊 REAL-TIME INGESTION PROGRESS MONITOR")
    print("=" * 70)
    
    checkpoint_path = os.getenv("GRAPH_CHECKPOINT", "data/ingestion/graph_checkpoint.json")
    prev_nodes = 0
    prev_chunks = 0
    iteration = 0
    
    try:
        while True:
            if max_iterations and iteration >= max_iterations:
                break
            
            iteration += 1
            checkpoint = _load_checkpoint(checkpoint_path)
            
            try:
                driver = get_neo4j_driver()
                with driver.session() as session:
                    result = session.run("MATCH (n) RETURN count(n) as count")
                    node_count = result.single()["count"]
                    result = session.run("MATCH ()-[r]-() RETURN count(r) as count")
                    rel_count = result.single()["count"]
                driver.close()
            except:
                node_count = 0
                rel_count = 0
            
            if checkpoint:
                last_idx = checkpoint.get("last_processed_index", -1)
                total = checkpoint.get("total_chunks", 0)
                progress = (last_idx + 1) / total * 100 if total > 0 else 0
                
                nodes_added = node_count - prev_nodes
                chunks_added = last_idx - prev_chunks
                
                print(f"\n[{time.strftime('%H:%M:%S')}] Iteration #{iteration}")
                print(f"  📈 Progress: {last_idx + 1}/{total} chunks ({progress:.1f}%)")
                print(f"  📦 Nodes: {node_count} ({'+' if nodes_added > 0 else ''}{nodes_added} new)")
                print(f"  🔗 Relationships: {rel_count}")
                
                if progress >= 100:
                    print("\n✅ INGESTION COMPLETE!")
                    break
                
                prev_nodes = node_count
                prev_chunks = last_idx
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Waiting for ingestion to start...")
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\n\n⏸️  Monitor stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        print("\n" + "=" * 70)

if __name__ == "__main__":
    monitor_progress(interval=10)
