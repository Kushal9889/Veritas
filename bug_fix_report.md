# Veritas RAG Stability Repair Report

## Scope and constraints
- Objective: repair breakages in the existing agentic RAG workflow without changing product behavior or adding new features.
- Focus areas: ingestion stability, retrieval reliability, orchestration compatibility, and LangGraph/LangSmith flow continuity.

## Reproduced failures before fixes
1. **Pytest collection hard-fail**
   - Command: `.venv/bin/python -m pytest -q`
   - Error: `ModuleNotFoundError: No module named 'src.orchestration.router'`
   - Impact: test/runtime compatibility broken after module consolidation.

2. **Latent runtime compatibility drifts**
   - Legacy import paths referenced modules removed/renamed in current architecture.
   - SSE source extraction in API expected old source string format and missed current retrieval output format.

3. **Ingestion/retrieval reliability defects**
   - Chroma recreate flag could be ignored due cached singleton lifecycle.
   - Retry-after parser regex was incorrectly escaped and could never parse numeric backoff.
   - Checkpoint writer failed when path had no directory component.
   - Retrieval parser could crash if Chroma metadata entry was `None`.
   - Gemini query expansion used incorrect SDK import/client path.

---

## Fixes applied (surgical)

### 1) Restored backward-compatible orchestration imports
**Files:** `src/orchestration/router.py`, `src/orchestration/nodes.py`, `src/ingestion/chunking.py`

- Added compatibility shim modules that re-export current workflow/chunker functions.
- This preserves consolidated architecture while restoring callers/tests that still use legacy module paths.

Corrected segments:
```python
# src/orchestration/router.py
from src.orchestration.workflow import _keyword_classify, build_metadata_filter, classify_query
```
```python
# src/orchestration/nodes.py
from src.orchestration.workflow import critic_node, generate_node, get_persona, retrieve_node
```
```python
# src/ingestion/chunking.py
from src.ingestion.chunkers import classify_chunk_section
```

### 2) Fixed Chroma recreation semantics
**File:** `src/core/caching.py:96`

- Root cause: `recreate=True` was ineffective when `_chroma_collection` was already cached.
- Fix: always process delete on recreate and reset cached collection to force a clean `get_or_create_collection`.

Corrected segment:
```python
if recreate:
    client.delete_collection(name=collection_name)
    _chroma_collection = None
```

### 3) Fixed retry-after parsing for graph extraction backoff
**File:** `src/ingestion/storage.py:138`

- Root cause: regex used double-escaped digit tokens (`\\d`) inside raw strings, so numeric wait extraction failed.
- Fix: corrected patterns to `(\d+)` and `\s`.

Corrected segment:
```python
match = re.search(r"retry in (\d+)", message, flags=re.IGNORECASE)
match = re.search(r"retry_after[:=\s]+(\d+)", message, flags=re.IGNORECASE)
```

### 4) Hardened checkpoint writes for basename paths
**File:** `src/ingestion/storage.py:165`

- Root cause: `os.makedirs(os.path.dirname(path), exist_ok=True)` fails when `dirname == ""`.
- Fix: only create directories when a directory component exists.

Corrected segment:
```python
directory = os.path.dirname(path)
if directory:
    os.makedirs(directory, exist_ok=True)
```

### 5) Prevented retrieval parse crash on null metadata
**File:** `src/retrieval/search.py:113`

- Root cause: candidate parsing assumed metadata object always present.
- Fix: normalize null metadata to `{}` before field access.

Corrected segment:
```python
metadata = search_results["metadatas"][0][idx] or {}
```

### 6) Corrected Gemini expansion client usage
**File:** `src/retrieval/search.py:185`

- Root cause: query expansion used an incompatible SDK import/client path.
- Fix: switched to `from google import genai` and `genai.Client(...)`, aligned with other modules.

Corrected segment:
```python
from google import genai
client = genai.Client(api_key=settings.GOOGLE_API_KEY)
```

### 7) Fixed SSE source extraction format mismatch
**File:** `src/api/server.py:75`

- Root cause: regex only matched legacy `(Source: ..., Page N)` format while retrieval emits bracketed refs (`[file pN]`).
- Fix: support both source formats.

Corrected segment:
```python
match = re.search(r"^\[[^\]]+\]", doc)
if not match:
    match = re.search(r"\(Source:.*?, Page \d+\)", doc)
```

### 8) Updated legacy Flask entrypoint to current modules
**File:** `app.py:11`

- Root cause: stale imports referenced deleted modules (`src.orchestration.graph`, `src.retrieval.reranker`).
- Fix: aligned imports to current workflow/search modules.

Corrected segment:
```python
from src.orchestration.workflow import build_graph
from src.retrieval.search import warmup as warmup_reranker
```

### 9) Restored Flask runtime dependency for legacy API entrypoint
**File:** `requirements.txt:6`

- Root cause: `app.py` depends on Flask, but dependency list omitted Flask.
- Fix: added pinned Flask dependency consistent with current stack pinning.

Corrected segment:
```text
flask==3.0.3
```

---

## Validation evidence after fixes
1. **Unit/integration smoke tests**
   - Command: `.venv/bin/python -m pytest -q`
   - Result: `7 passed` (no collection/runtime failures)

2. **Syntax/bytecode compile check**
   - Command: `.venv/bin/python -m compileall -q main.py src tests app.py`
   - Result: pass

3. **CLI workflow runtime check**
   - Command: `.venv/bin/python main.py query "hi"`
   - Result: end-to-end graph execution completed with expected offline fallbacks.

## Residual non-code environment constraints (not defects in patched logic)
- External network is restricted in this execution environment, so LangSmith and remote model endpoints log connection warnings.
- These warnings do not prevent local fallback execution paths from completing.
