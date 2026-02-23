"""Production-grade cache — Redis with in-memory LRU fallback.
Includes ChromaDB client/collection singletons."""

import redis
import json
import hashlib
from typing import Any, Optional
import os
from collections import OrderedDict

# Redis connection
_redis_client = None

def get_redis_client():
    """Get or create Redis client"""
    global _redis_client
    if _redis_client is None:
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            _redis_client = redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=2)
            _redis_client.ping()
        except Exception:
            _redis_client = None
    return _redis_client

def cache_get(key: str) -> Optional[Any]:
    try:
        client = get_redis_client()
        if client is None:
            return None
        value = client.get(key)
        return json.loads(value) if value else None
    except Exception:
        return None

def cache_set(key: str, value: Any, ttl: int = 3600):
    try:
        client = get_redis_client()
        if client is None:
            return
        client.setex(key, ttl, json.dumps(value))
    except Exception:
        pass

def cache_key(prefix: str, *args) -> str:
    key_str = f"{prefix}:" + ":".join(str(arg) for arg in args)
    return hashlib.md5(key_str.encode()).hexdigest()

# In-memory LRU fallback
_embedding_cache = OrderedDict()
MAX_CACHE_SIZE = 1000

def get_cached_embedding(text: str):
    """Get cached embedding from Redis or LRU."""
    key = cache_key("embedding", text)
    redis_val = cache_get(key)
    if redis_val is not None:
        return redis_val
        
    lru_key = hashlib.md5(text.encode()).hexdigest()
    if lru_key in _embedding_cache:
        _embedding_cache.move_to_end(lru_key)
        return _embedding_cache[lru_key]
    return None

def cache_embedding(text: str, embedding):
    """Cache embedding in Redis and LRU."""
    key = cache_key("embedding", text)
    cache_set(key, embedding, ttl=86400)
    
    lru_key = hashlib.md5(text.encode()).hexdigest()
    _embedding_cache[lru_key] = embedding
    _embedding_cache.move_to_end(lru_key)
    while len(_embedding_cache) > MAX_CACHE_SIZE:
        _embedding_cache.popitem(last=False)

def get_cached_graph_path(entity: str):
    key = cache_key("graph_path", entity.lower())
    return cache_get(key)

def cache_graph_path(entity: str, results):
    key = cache_key("graph_path", entity.lower())
    cache_set(key, results, ttl=3600)

_chroma_client = None
_chroma_collection = None

def get_cached_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        from src.core.config import settings
        _chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
    return _chroma_client

def get_cached_chroma_collection(collection_name: str, embedding_fn=None, recreate: bool = False):
    global _chroma_collection
    client = get_cached_chroma_client()

    if recreate:
        try:
            client.delete_collection(name=collection_name)
        except ValueError:
            pass
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Failed to delete collection: {e}")
        _chroma_collection = None

    if _chroma_collection is None:
        _chroma_collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
    return _chroma_collection

