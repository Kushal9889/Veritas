"""
Production Retrieval Pipeline (Consolidated)
Handles Dense (Chroma), Graph (Neo4j), and Reranking operations.
"""
import sys
import re
import json
import time
import requests
from typing import List, Dict, Tuple

from langchain_core.runnables import RunnableLambda
from langchain_community.embeddings import OllamaEmbeddings as LCOllamaEmbeddings
from langchain_community.graphs import Neo4jGraph
from sentence_transformers import CrossEncoder
from langsmith import traceable

from src.core.config import settings, get_gemini_client_with_fallback
from src.core.telemetry import get_telemetry
from src.core.caching import (
    get_cached_embedding, 
    cache_embedding, 
    get_cached_chroma_collection,
    get_cached_graph_path, 
    cache_graph_path
)

telemetry = get_telemetry('retrieval.search')

# ── 1. Centralized Embeddings ──────────────────────────────────────────
if settings.EMBEDDING_PROVIDER == "gemini":
    # Lazy import: langchain_google_genai has a pydantic v1/v2 metaclass conflict
    # that only surfaces at import time — guard it here so Ollama paths never crash.
    from langchain_google_genai import GoogleGenerativeAIEmbeddings  # noqa: E402
    _embedder = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=settings.GOOGLE_API_KEY
    )
else:
    _embedder = LCOllamaEmbeddings(
        model=settings.OLLAMA_EMBEDDING_MODEL,
        base_url=settings.OLLAMA_BASE_URL
    )

def _embed_query(query_text: str):
    """Embed query with caching."""
    cached = get_cached_embedding(query_text)
    if cached:
        return cached
    vector = _embedder.embed_query(query_text)
    cache_embedding(query_text, vector)
    return vector

# ── 2. Cross-Encoder Reranking ─────────────────────────────────────────
class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize cross-encoder model."""
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """Rerank documents using cross-encoder scoring."""
        if not documents:
            return []
        pairs = [[query, doc.get('text', doc.get('child_text', ''))] for doc in documents]
        scores = self.model.predict(pairs)
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
        reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        return reranked[:top_k]

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker

def warmup():
    """Pre-load the reranker model at app startup to avoid cold-start latency."""
    reranker = get_reranker()
    reranker.model.predict([["warmup query", "warmup document"]])
    return True

# ── 3. Dense Retrieval (ChromaDB) ──────────────────────────────────────
_parent_map = None
def load_parent_map():
    global _parent_map
    if _parent_map is None:
        try:
            with open('data/cache/parent_map.json', 'r') as f:
                _parent_map = json.load(f)
        except FileNotFoundError:
            _parent_map = {}
    return _parent_map

def _search_chroma(query_vector: list, top_k: int, metadata_filter: dict = None):
    """Search ChromaDB using HNSW topology."""
    collection = get_cached_chroma_collection(settings.CHROMA_COLLECTION_NAME)
    kwargs = dict(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["metadatas", "distances"]
    )
    if metadata_filter:
        kwargs["where"] = metadata_filter
    return collection.query(**kwargs)

def _parse_results(search_results: dict, min_similarity: float = 70.0):
    """Parse ChromaDB results into candidate dicts and filter by absolute similarity bounds."""
    candidates = []
    if not search_results or not search_results.get("ids") or not search_results["ids"][0]:
        return candidates

    for idx in range(len(search_results["ids"][0])):
        doc_id = search_results["ids"][0][idx]
        metadata = search_results["metadatas"][0][idx] or {}
        distance = search_results["distances"][0][idx]
        
        # Invert for percentage (assuming cosine distance 0-2)
        similarity_score = (1.0 - (distance / 2.0)) * 100.0

        if similarity_score < min_similarity:
            continue

        candidates.append({
            "target_id": doc_id,
            "exact_metadata": metadata,
            "similarity_score_percent": round(similarity_score, 2),
            "id": doc_id,
            "child_text": metadata.get('text', 'No text found.'),
            "text": metadata.get('text', 'No text found.'),
            "child_id": metadata.get('child_id', ''),
            "parent_id": metadata.get('parent_id', ''),
            "score": similarity_score / 100.0,
            "source": metadata.get('source', 'Unknown File').split("/")[-1] if isinstance(metadata.get('source'), str) else 'Unknown',
            "page": int(float(metadata.get('page', 0))) + 1 if str(metadata.get('page', '0')).replace('.','',1).isdigit() else 1,
            "section": metadata.get('section', 'Unknown'),
            "collection_id": metadata.get('collection_id', 'Unknown'),
            "ticker": metadata.get('ticker', 'Unknown'),
            "year": metadata.get('year', 'Unknown')
        })
    return candidates

def _rerank_and_fetch_parents(query_text: str, candidates: list, top_k: int):
    """Rerank candidates → fetch parent contexts → deduplicate by parent_id."""
    reranker = get_reranker()
    reranked = reranker.rerank(
        query_text,
        [{'text': c['child_text'], **c} for c in candidates],
        top_k=top_k + 3
    )

    parent_map = load_parent_map()
    seen_parents = set()
    unique_results = []
    for doc in reranked:
        parent_id = doc.get('parent_id', '')
        if parent_id and parent_id in seen_parents:
            continue
        if parent_id:
            seen_parents.add(parent_id)
        doc['text'] = parent_map.get(doc.get('child_id', ''), doc['child_text'])
        unique_results.append(doc)

    return unique_results[:top_k]

def _boost_by_keywords(candidates: list, keywords: list) -> list:
    """Post-rerank relevance boost for chunks matching router keywords."""
    if not keywords:
        return candidates
    for doc in candidates:
        text_lower = doc.get('text', doc.get('child_text', '')).lower()
        matches = sum(1 for kw in keywords if kw.lower() in text_lower)
        doc['keyword_boost'] = matches * 0.5
    candidates.sort(
        key=lambda x: x.get('rerank_score', x.get('score', 0)) + x.get('keyword_boost', 0),
        reverse=True
    )
    return candidates

def expand_query(query_text: str) -> str:
    """Expand user query into a statistically optimal dense representation."""
    prompt = f"""You are a query expansion system for an enterprise financial RAG pipeline analyzing SEC 10-K filings.
Take the user's short query and expand it with relevant business, financial, and operational keywords to maximize vector retrieval relevance.
Output ONLY the expanded query string. No quotes, no filler text, no conversational responses.
User Query: {query_text}"""
    try:
        if settings.EMBEDDING_PROVIDER == "gemini":
            client = get_gemini_client_with_fallback()
            for attempt in range(len(settings.GOOGLE_API_KEYS)):
                try:
                    res = client.models.generate_content(model=settings.GENERATION_MODEL, contents=prompt)
                    expanded = res.text.strip()
                    if expanded:
                        telemetry.log_info("Query expanded successfully (Gemini)", original=query_text, expanded=expanded)
                        return expanded
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower():
                        settings.get_next_api_key()
                        client = get_gemini_client_with_fallback()
                        time.sleep(2)
                    else:
                        raise
        else:
            res = requests.post(f"{settings.OLLAMA_BASE_URL}/api/generate", json={
                "model": settings.ROUTER_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 60}
            }, timeout=3)
            if res.status_code == 200:
                expanded = res.json().get("response", "").strip()
                if expanded:
                    telemetry.log_info("Query expanded successfully (Ollama)", original=query_text, expanded=expanded)
                    return expanded
    except Exception as e:
        telemetry.log_warning("Query expansion failed, using original", error=str(e))
    return query_text

@traceable(run_type="retriever", name="Hybrid_Financial_Search")
def query_financial_docs(query_text: str, top_k: int = 5, use_reranking: bool = True,
                         metadata_filter: dict = None, router_keywords: list = None, 
                         strict_filter: bool = False, min_similarity: float = 70.0,
                         expand: bool = True):
    """Production retrieval: expand → embed → strict bound search → rerank → boost → dedup."""
    try:
        final_query = expand_query(query_text) if expand else query_text
        embed_chain = RunnableLambda(lambda q: _embed_query(q))
        fetch_k = top_k * 10 if use_reranking else top_k

        def _search_with_fallback(vec):
            if metadata_filter:
                results = _search_chroma(vec, fetch_k, metadata_filter)
                if results and results.get('ids') and results['ids'][0]:
                    return results
                if strict_filter:
                    telemetry.log_info("Strict filter returned 0 results. Returning empty.", filter=str(metadata_filter))
                    return {"ids": [[]], "metadatas": [[]], "distances": [[]]}
                telemetry.log_info("Filter returned 0 results, retrying unfiltered", filter=str(metadata_filter))
            return _search_chroma(vec, fetch_k)

        search_chain = RunnableLambda(_search_with_fallback)
        parse_chain = RunnableLambda(lambda results: _parse_results(results, min_similarity))

        retrieval_chain = embed_chain | search_chain | parse_chain
        candidates = retrieval_chain.invoke(final_query)

        if not candidates:
            telemetry.log_warning("Zero candidates met the strict commercial similarity threshold", min_similarity=min_similarity)
            return []

        if use_reranking and candidates:
            reranked = _rerank_and_fetch_parents(query_text, candidates, top_k)
            if router_keywords:
                reranked = _boost_by_keywords(reranked, router_keywords)
            return reranked

        parent_map = load_parent_map()
        for doc in candidates[:top_k]:
            doc['text'] = parent_map.get(doc.get('child_id', ''), doc['child_text'])
        return candidates[:top_k]

    except Exception as e:
        telemetry.log_error("Retrieval failed", error=str(e))
        return []

# ── 4. Graph Retrieval (Neo4j) ─────────────────────────────────────────
def get_neo4j_driver():
    from neo4j import GraphDatabase
    return GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
    )

def _query_with_cache(entity_name: str):
    """Query graph with Redis caching and parameterized Cypher (no injection).
    
    Matches against both n.id (snake_case normalized, e.g. 'apple_inc') and
    n.name (display form, e.g. 'Apple Inc.') so natural-language entity strings
    from the router reliably hit graph nodes regardless of normalization.
    """
    if not entity_name or not entity_name.strip():
        return []
    entity_name = entity_name.strip()
    cached = get_cached_graph_path(entity_name)
    if cached is not None:
        return cached

    # Normalize to snake_case for id matching (mirrors extractor normalization)
    import re as _re
    entity_normalized = _re.sub(r"[^\w\s]", "", entity_name.lower().strip())
    entity_normalized = _re.sub(r"\s+", "_", entity_normalized).strip("_")

    query = """
    MATCH (n)-[r]->(m)
    WHERE toLower(n.id) CONTAINS toLower($entity_norm)
       OR toLower(n.name) CONTAINS toLower($entity_raw)
    RETURN labels(n)[1] AS lA, n.id AS source, n.name AS source_name,
           type(r) AS relationship,
           labels(m)[1] AS lB, m.id AS target, m.name AS target_name
    LIMIT 10
    """

    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            records = session.run(query, entity_norm=entity_normalized, entity_raw=entity_name)
            results = []
            for r in records:
                src_label = r["lA"] or "Entity"
                tgt_label = r["lB"] or "Entity"
                # Use display name when available, fall back to id
                src_display = r["source_name"] or r["source"] or ""
                tgt_display = r["target_name"] or r["target"] or ""
                results.append({
                    "source": f"({src_display}:{src_label})",
                    "relationship": r["relationship"],
                    "target": f"({tgt_display}:{tgt_label})",
                })

        cache_graph_path(entity_name, results)
        return results
    except Exception as e:
        telemetry.log_warning("Graph query failed", entity=entity_name, error=str(e))
        return []

graph_query_chain = RunnableLambda(_query_with_cache)

def query_graph(entity_name: str):
    """LangChain-based graph query with caching."""
    return graph_query_chain.invoke(entity_name)



if __name__ == "__main__":
    if len(sys.argv) > 1:
        res = query_financial_docs(" ".join(sys.argv[1:]))
        for r in res:
            print(r.get("text", "")[:100])
