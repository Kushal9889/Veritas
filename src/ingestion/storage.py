"""
Unified Storage Module for the Ingestion Pipeline.
Handles Vector databases (ChromaDB) and Graph databases (Neo4j).
"""
import os
import time
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Any

from google import genai

from src.core.config import settings, get_gemini_client_with_fallback, NODE_LABELS, REL_TYPES
from src.core.caching import get_cached_chroma_collection
from src.core.models import OllamaEmbeddings
from src.core.telemetry import get_telemetry

telemetry = get_telemetry('ingestion.storage')
logger = logging.getLogger('ingestion.storage')

# ── 1. Vector DB Ingestion (ChromaDB) ──────────────────────────────────
def get_embedder():
    if settings.EMBEDDING_PROVIDER == "ollama":
        return OllamaEmbeddings(model=settings.OLLAMA_EMBEDDING_MODEL, base_url=settings.OLLAMA_BASE_URL)
    else:
        return get_gemini_client_with_fallback()

def embed_with_retry(texts, embedder, max_retries=3):
    if settings.EMBEDDING_PROVIDER == "ollama":
        return embedder.embed_documents(texts)

    client = embedder
    for attempt in range(max_retries * len(settings.GOOGLE_API_KEYS)):
        try:
            response = client.models.embed_content(
                model="text-embedding-004",  
                contents=texts,
                config={"task_type": "RETRIEVAL_DOCUMENT", "output_dimensionality": 768}
            )
            return [e.values for e in response.embeddings]
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                settings.get_next_api_key()
                client = get_gemini_client_with_fallback()
                import re
                match = re.search(r'retry in (\d+)', error_msg)
                wait_time = int(match.group(1)) + 1 if match else 2
                print(f"   ⏳ Rate limit hit. Switching key and waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    print("   ❌ All API keys exhausted.")
    return None

def _process_batch(batch_data, embedder, collection, batch_num, total_batches):
    batch_items, text_fields = batch_data
    docs_to_embed = []
    enriched_metadatas = []
    
    for item in batch_items:
        extracted_texts = [str(item.get(field, "")) for field in text_fields if item.get(field)]
        dense_document = " ".join(extracted_texts).strip()
        
        meta = {k: v for k, v in item.items() if k not in text_fields}
        meta["text"] = dense_document
        
        docs_to_embed.append(dense_document)
        enriched_metadatas.append(meta)

    try:
        vectors = embed_with_retry(docs_to_embed, embedder)
        if vectors is None:
            return 0
            
        valid_docs, valid_metas, valid_vecs, ids = [], [], [], []

        for doc, vector, meta in zip(docs_to_embed, vectors, enriched_metadatas):
            if not doc:
                continue
            hash_input = (doc + str(meta)).encode('utf-8')
            unique_id = hashlib.sha256(hash_input).hexdigest()
            valid_docs.append(doc)
            valid_metas.append(meta)
            valid_vecs.append(vector)
            ids.append(unique_id)
            
        if ids:
            collection.add(embeddings=valid_vecs, documents=valid_docs, metadatas=valid_metas, ids=ids)
            print(f"   ✅ Batch {batch_num}/{total_batches} uploaded (Parallel).")
        return 1
    except Exception as e:
        print(f"   ❌ Batch {batch_num} failed: {e}")
        return 0

def ingest_documents(raw_data: List[Dict], text_fields: List[str], recreate_collection: bool = False):
    if not raw_data:
        telemetry.log_warning("No documents provided for ingestion.")
        return

    print(f"🔌 Connecting to ChromaDB Collection: {settings.CHROMA_COLLECTION_NAME}")
    print(f"🧠 Using {settings.EMBEDDING_PROVIDER.upper()} embeddings")

    try:
        embedder = get_embedder()
        collection = get_cached_chroma_collection(settings.CHROMA_COLLECTION_NAME, recreate=recreate_collection)

        batch_size = 100 
        total_batches = (len(raw_data) + batch_size - 1) // batch_size
        collections = set(d.get("collection_id", "UNKNOWN") for d in raw_data)
        sections = set(d.get("section", "UNKNOWN") for d in raw_data)
        
        print(f"🚀 Starting PARALLEL ingestion of {len(raw_data)} chunks ({total_batches} batches)...")
        print(f"   📚 Collections: {collections}\n   📑 Sections found: {len(sections)} unique sections")

        successful_batches = 0
        total_start = time.time()
        
        batches = [(raw_data[i : i + batch_size], text_fields) for i in range(0, len(raw_data), batch_size)]

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(_process_batch, batch, embedder, collection, i+1, total_batches) for i, batch in enumerate(batches)]
            for future in as_completed(futures):
                successful_batches += future.result()

        total_duration = time.time() - total_start
        telemetry.log_info("Ingestion completed", successful_batches=successful_batches, total_batches=total_batches, duration_s=round(total_duration, 2))
        print(f"🎉 Ingestion Complete: {successful_batches}/{total_batches} batches in {total_duration:.1f}s")
    except Exception as e:
        telemetry.log_error(f"Critical failure during vector database ingestion: {e}")
        print(f"❌ Aborting ingestion sequence due to error: {e}")
        raise

# ── 2. Graph Database Ingestion (Neo4j) ────────────────────────────────

# ─── Neo4j Connection ──────────────────────────────────────────────────

_neo4j_driver = None

def get_neo4j_driver():
    """Get or create a singleton Neo4j driver.
    
    Returns:
        Neo4j driver instance.
    """
    global _neo4j_driver
    if _neo4j_driver is None:
        from neo4j import GraphDatabase
        _neo4j_driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        )
    return _neo4j_driver


# ─── Schema Hygiene & Audit ────────────────────────────────────────────

def audit_and_prepare_schema() -> None:
    """Audit the existing graph and enforce clean schema state.
    
    Steps:
      1. Drop orphaned nodes (nodes with no relationships and no useful data).
      2. Create UNIQUE constraints on Entity.id.
      3. Create indexes for fast lookups.
      
    This runs before every ingestion to guarantee schema correctness.
    """
    import logging as _log
    _log.getLogger("neo4j").setLevel(_log.ERROR)

    driver = get_neo4j_driver()
    logger.info("Starting graph schema audit and preparation")

    with driver.session() as session:
        # 1. Count current state
        result = session.run("MATCH (n) RETURN count(n) AS cnt")
        node_count = result.single()["cnt"]
        result = session.run("MATCH ()-[r]-() RETURN count(r) AS cnt")
        rel_count = result.single()["cnt"]
        logger.info(f"Current graph state: {node_count} nodes, {rel_count} relationships")
        print(f"   📊 Pre-audit state: {node_count} nodes, {rel_count} relationships")

        # 2. Remove orphaned nodes (no relationships, empty id)
        result = session.run(
            """MATCH (n:Entity)
               WHERE NOT (n)--() AND (n.id IS NULL OR n.id = '')
               DELETE n
               RETURN count(n) AS deleted"""
        )
        deleted = result.single()["deleted"]
        if deleted > 0:
            logger.info(f"Removed {deleted} orphaned nodes")
            print(f"   🧹 Removed {deleted} orphaned nodes")

        # 3. Create constraints and indexes
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
        ]
        for stmt in constraints:
            try:
                session.run(stmt)
            except Exception as e:
                logger.debug(f"Constraint may already exist: {e}")

        # 4. Create indexes for each node label for fast MERGE
        # Covers all 15+ entity types in the expanded ontology
        for label in NODE_LABELS:
            try:
                session.run(
                    f"CREATE INDEX IF NOT EXISTS FOR (n:`{label}`) ON (n.id)"
                )
            except Exception as e:
                logger.debug(f"Index for {label} may already exist: {e}")

        # 5. Create composite index for name lookups
        try:
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.name)")
        except Exception as e:
            logger.debug(f"Name index may already exist: {e}")

    print("   ✅ Schema audit complete. Constraints and indexes enforced.")
    logger.info("Schema audit complete")


def clear_graph_database() -> None:
    """Wipe all nodes and relationships from Neo4j."""
    import logging as _log
    _log.getLogger("neo4j").setLevel(_log.ERROR)

    print("🧹 Wiping Neo4j AuraDB graph completely...")
    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            # Batch delete to avoid memory issues on large graphs
            while True:
                result = session.run(
                    "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(n) AS deleted"
                )
                deleted = result.single()["deleted"]
                if deleted == 0:
                    break
                logger.info(f"Deleted batch of {deleted} nodes")
        print("✅ Graph wiped successfully.")
    except Exception as e:
        print(f"❌ Failed to wipe graph: {e}")
        logger.error(f"Graph wipe failed: {e}")


# ─── Batch UNWIND Ingestion ────────────────────────────────────────────

class Neo4jBatchWriter:
    """High-performance batch writer using Cypher UNWIND.
    
    Accumulates nodes and edges, then flushes them in bulk using
    UNWIND statements for maximum throughput. Each flush is a single
    atomic network request.
    
    Usage:
        writer = Neo4jBatchWriter(batch_size=50)
        writer.add_nodes([{"id": "Tesla", "type": "Company"}, ...])
        writer.add_edges([{"source": "Tesla", "target": "Risk", "type": "EXPOSED_TO_RISK"}, ...])
        writer.flush()  # commits everything
        stats = writer.get_stats()
    """

    def __init__(self, batch_size: int = 50) -> None:
        """Initialize batch writer.
        
        Args:
            batch_size: Number of items to accumulate before auto-flush.
        """
        self._driver = get_neo4j_driver()
        self._batch_size = batch_size
        self._node_buffer: List[Dict[str, str]] = []
        self._edge_buffer: List[Dict[str, str]] = []
        self._total_nodes_written: int = 0
        self._total_edges_written: int = 0
        self._flush_count: int = 0

    def add_nodes(self, nodes: List[Dict[str, str]]) -> None:
        """Add nodes to the buffer. Auto-flushes when batch_size reached.
        
        Args:
            nodes: List of node dicts with 'id' and 'type' keys.
        """
        self._node_buffer.extend(nodes)
        if len(self._node_buffer) >= self._batch_size:
            self._flush_nodes()

    def add_edges(self, edges: List[Dict[str, str]]) -> None:
        """Add edges to the buffer. Auto-flushes when batch_size reached.
        
        Args:
            edges: List of edge dicts with 'source', 'target', 'type' keys.
        """
        self._edge_buffer.extend(edges)
        if len(self._edge_buffer) >= self._batch_size:
            self._flush_edges()

    def _flush_nodes(self) -> None:
        """Commit buffered nodes using UNWIND for batch efficiency.
        
        Uses normalized entity_id for deterministic MERGE to prevent
        duplicate nodes. Entity names like "Apple Inc.", "Apple", "AAPL"
        all resolve to the same node via their normalized id.
        """
        if not self._node_buffer:
            return
        try:
            with self._driver.session() as session:
                # Group nodes by type for labeled MERGE
                by_type: Dict[str, List[Dict[str, str]]] = {}
                for node in self._node_buffer:
                    n_type = node.get("type", "Entity")
                    n_id = node.get("id", "").strip()
                    n_name = node.get("name", n_id).strip()
                    if n_id:
                        sanitized = "".join(c for c in n_type if c.isalnum()) or "Entity"
                        by_type.setdefault(sanitized, []).append(
                            {"id": n_id, "name": n_name}
                        )

                for n_type, items in by_type.items():
                    # Step 1: MERGE on Entity.id (the constrained label) — never CREATE.
                    # This guarantees idempotency even if the same id arrives from
                    # multiple chunks with different declared types.
                    session.run(
                        """UNWIND $items AS item
                           MERGE (n:Entity {id: item.id})
                           ON CREATE SET n.name = item.name
                           ON MATCH  SET n.name = COALESCE(n.name, item.name)""",
                        items=items
                    )
                    # Step 2: SET the specific label as an additional label.
                    # SET n:Product does NOT fail if the label already exists.
                    if n_type != "Entity":
                        session.run(
                            f"""UNWIND $items AS item
                                MATCH (n:Entity {{id: item.id}})
                                SET n:`{n_type}`""",
                            items=items
                        )
                self._total_nodes_written += len(self._node_buffer)
                self._flush_count += 1
        except Exception as e:
            logger.error(f"Node flush failed: {e}")
        finally:
            self._node_buffer.clear()

    def _flush_edges(self) -> None:
        """Commit buffered edges using UNWIND for batch efficiency.
        
        Uses MERGE for idempotent relationship creation.
        """
        if not self._edge_buffer:
            return
        try:
            with self._driver.session() as session:
                # Group edges by relationship type
                by_rel: Dict[str, List[Dict[str, str]]] = {}
                for edge in self._edge_buffer:
                    rel = edge.get("type", "RELATED_TO")
                    sanitized = "".join(
                        c for c in rel if c.isalnum() or c == "_"
                    ).upper() or "RELATED_TO"
                    by_rel.setdefault(sanitized, []).append({
                        "source": edge["source"].strip(),
                        "target": edge["target"].strip(),
                    })

                for rel_type, pairs in by_rel.items():
                    session.run(
                        f"""UNWIND $pairs AS pair
                            MATCH (s:Entity {{id: pair.source}})
                            MATCH (t:Entity {{id: pair.target}})
                            MERGE (s)-[:`{rel_type}`]->(t)""",
                        pairs=pairs
                    )
                self._total_edges_written += len(self._edge_buffer)
        except Exception as e:
            logger.error(f"Edge flush failed: {e}")
        finally:
            self._edge_buffer.clear()

    def flush(self) -> None:
        """Flush all remaining buffered data to Neo4j."""
        self._flush_nodes()
        self._flush_edges()

    def get_stats(self) -> Dict[str, int]:
        """Get write statistics.
        
        Returns:
            Dict with total_nodes, total_edges, flush_count.
        """
        return {
            "total_nodes": self._total_nodes_written,
            "total_edges": self._total_edges_written,
            "flush_count": self._flush_count,
        }


# ─── Legacy Compatibility ─────────────────────────────────────────────
# These are kept for backward compatibility with existing imports.

def _load_checkpoint(path: str) -> Optional[dict]:
    """Load JSON checkpoint (legacy)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _write_checkpoint(path: str, payload: dict) -> None:
    """Write JSON checkpoint atomically (legacy)."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

def create_graph_indexes():
    """Create Neo4j indexes (legacy wrapper)."""
    audit_and_prepare_schema()


# ─── Production Ingestion Orchestrator ─────────────────────────────────

def ingest_into_graph(
    documents,
    clear: bool = True,
    *,
    resume_from_checkpoint: bool = False,
    checkpoint_path: Optional[str] = "data/ingestion/graph_checkpoint.json",
    start_index: int = 0,
) -> None:
    """Production graph ingestion with checkpoint tracking and batch writes.
    
    Uses the new extractor (circuit breaker + Ollama fallback) and
    Neo4jBatchWriter (UNWIND) for maximum throughput.
    
    Args:
        documents: List of LangChain Document objects.
        clear: Whether to wipe the graph before ingestion.
        resume_from_checkpoint: Whether to resume from last checkpoint.
        checkpoint_path: Path to JSON checkpoint file.
        start_index: Starting chunk index (overridden by checkpoint).
    """
    from src.ingestion.extractor import extract_graph
    from src.ingestion.checkpoint import (
        CheckpointManager, compute_content_hash
    )

    total = len(documents)
    if total == 0:
        logger.warning("No documents provided for graph ingestion")
        return

    # Initialize checkpoint manager
    doc_hash = compute_content_hash([d.page_content[:200] for d in documents])
    ckpt_mgr = CheckpointManager(settings.CHECKPOINT_DB_PATH)
    state = ckpt_mgr.initialize_or_resume(doc_hash, total)

    if state.last_processed_index >= 0:
        start_index = state.last_processed_index + 1
        print(f"🔁 Resuming from chunk {start_index + 1}/{total} "
              f"({state.success_count} success, {state.failed_count} failed)")
    elif clear:
        clear_graph_database()

    # Schema audit
    audit_and_prepare_schema()

    # Batch writer
    writer = Neo4jBatchWriter(batch_size=settings.NEO4J_BATCH_SIZE)

    print(f"--- 🕸️ GRAPH: Processing {total - start_index} remaining chunks "
          f"(of {total} total) to AuraDB ---")

    ingestion_start = time.time()

    for i in range(start_index, total):
        # Skip already processed chunks
        if ckpt_mgr.is_chunk_processed(i):
            continue

        chunk_start = time.time()
        doc = documents[i]
        text = doc.page_content

        progress_pct = ((i + 1) / total) * 100
        print(f"   [{i+1}/{total}] ({progress_pct:.1f}%) Processing...", end=" ", flush=True)

        ckpt_mgr.mark_in_progress(i)

        try:
            graph_data, model_used = extract_graph(text)
            nodes = graph_data.get("nodes", [])
            edges = graph_data.get("edges", [])
            elapsed_ms = (time.time() - chunk_start) * 1000

            if not nodes and not edges:
                ckpt_mgr.mark_skipped(i)
                print(f"⏭️  skipped (no entities) [{model_used}]")
                continue

            # Write to Neo4j via batch writer
            writer.add_nodes(nodes)
            writer.add_edges(edges)

            # Flush every N chunks to prevent unbounded memory
            if (i + 1) % settings.NEO4J_BATCH_SIZE == 0:
                writer.flush()

            ckpt_mgr.mark_success(
                i, nodes=len(nodes), edges=len(edges),
                model=model_used, time_ms=elapsed_ms
            )
            print(f"✅ {len(nodes)}N {len(edges)}E [{model_used}] ({elapsed_ms:.0f}ms)")

        except Exception as e:
            elapsed_ms = (time.time() - chunk_start) * 1000
            ckpt_mgr.mark_failed(i, error=str(e)[:200])
            logger.error(f"Chunk {i} failed: {e}")
            print(f"❌ failed: {str(e)[:80]}")

        # Write JSON checkpoint for backward compatibility
        if checkpoint_path and (i + 1) % 10 == 0:
            ckpt_mgr.write_json_checkpoint(checkpoint_path)

    # Final flush
    writer.flush()
    stats = writer.get_stats()

    # Finalize
    final_state = ckpt_mgr.finalize()
    if checkpoint_path:
        ckpt_mgr.write_json_checkpoint(checkpoint_path)

    total_time = time.time() - ingestion_start
    print(f"\n{'='*60}")
    print(f"🏁 Graph Ingestion Complete")
    print(f"   ⏱️  Duration: {total_time:.1f}s")
    print(f"   ✅ Success: {final_state.success_count}")
    print(f"   ❌ Failed:  {final_state.failed_count}")
    print(f"   ⏭️  Skipped: {final_state.skipped_count}")
    print(f"   📦 Nodes written: {stats['total_nodes']}")
    print(f"   🔗 Edges written: {stats['total_edges']}")
    print(f"   🔄 Batch flushes: {stats['flush_count']}")
    print(f"{'='*60}")

    telemetry.log_info(
        "Graph ingestion complete",
        duration_s=round(total_time, 2),
        success=final_state.success_count,
        failed=final_state.failed_count,
        skipped=final_state.skipped_count,
        nodes=stats["total_nodes"],
        edges=stats["total_edges"],
    )
