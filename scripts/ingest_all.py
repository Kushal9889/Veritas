"""
Batch ingestion of a large local corpus (10-K docs) into Chroma + Neo4j.

Production pipeline:
  1. Loads PDFs / TXT / MD from data/raw.
  2. Splits into 1000-char chunks.
  3. Ingests vectors into ChromaDB.
  4. Ingests knowledge graph into Neo4j AuraDB using:
     - ACID SQLite checkpointing (resume-safe)
     - Circuit-breaker LLM extraction (Groq → Gemini → Ollama)
     - UNWIND batch Neo4j writes
"""
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List

# ── Path bootstrap ─────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Shim for langchain_google_genai pydantic v1 compat
try:
    import pydantic.v1 as _pyd_v1  # type: ignore
    sys.modules.setdefault("langchain_core.pydantic_v1", _pyd_v1)
except Exception:
    pass

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.ingestion.storage import ingest_documents, ingest_into_graph
from src.ingestion.chunkers import classify_chunk_section
from src.core.config import settings
from src.core.telemetry import get_telemetry

telemetry = get_telemetry("scripts.ingest_all")

ROOT = Path(os.getenv("DOC_ROOT", "data/raw"))
BATCH_CHUNKS = int(os.getenv("BATCH_CHUNKS", "1000"))


# ── File Discovery ─────────────────────────────────────────────────────

def iter_files(root: Path) -> Iterable[Path]:
    """Yield all ingestible files from the data directory."""
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md"}:
            yield path


# ── Load & Chunk ───────────────────────────────────────────────────────

def load_and_chunk(path: Path, splitter: RecursiveCharacterTextSplitter) -> List[Document]:
    """Load a file and split into LangChain Document chunks."""
    try:
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path), autodetect_encoding=True)
        raw_docs = loader.load()
        for doc in raw_docs:
            doc.metadata.setdefault("source", path.name)
            doc.metadata.setdefault("section", "general")
        chunks = splitter.split_documents(raw_docs)
        # Tag each chunk with its classified section so metadata filters work at query time
        for chunk in chunks:
            chunk.metadata["section"] = classify_chunk_section(chunk.page_content)
        return chunks
    except Exception as e:
        telemetry.log_warning("Skipping file due to load error", file=str(path), error=str(e))
        return []


# ── Vector Prep ────────────────────────────────────────────────────────

def to_vector_dicts(docs: List[Document]) -> List[dict]:
    """Convert Document list to dict format for ChromaDB ingestion."""
    return [
        {
            "text": d.page_content,
            "page": d.metadata.get("page", d.metadata.get("page_num", 0)),
            "source": d.metadata.get("source", "unknown"),
            "section": d.metadata.get("section", "general"),
            "collection_id": d.metadata.get("collection_id", ROOT.name),
        }
        for d in docs
    ]


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    if not ROOT.exists():
        raise SystemExit(f"Data root not found: {ROOT}")

    groq_available = settings.get_available_groq_key_count()

    print("=" * 60)
    print("🚀 Veritas Production Ingestion Pipeline")
    print(f"   📂 Source: {ROOT}")
    print(f"   🧠 LLM Cascade: Groq({settings.GROQ_MODEL}) → "
          f"Groq({settings.GROQ_FALLBACK_MODEL}) → Gemini → Ollama")
    print(f"   📦 Neo4j batch size: {settings.NEO4J_BATCH_SIZE}")
    print(f"   🔑 Groq keys: {groq_available}/{len(settings.GROQ_API_KEYS)} available")
    print(f"   🔑 Gemini keys: {len(settings.GOOGLE_API_KEYS)}")
    print(f"   🏠 Ollama fallback: {settings.GRAPH_EXTRACTION_OLLAMA_MODEL}")
    print("=" * 60)

    telemetry.log_info("Starting bulk ingestion", root=str(ROOT))
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=120)

    checkpoint_path = os.getenv(
        "GRAPH_CHECKPOINT", str(settings.CHECKPOINT_JSON_PATH)
    )

    # ── 1. Collect all chunks across all files ──
    print("\n📄 Loading and chunking documents...")
    all_chunks: List[Document] = []
    for path in iter_files(ROOT):
        chunks = load_and_chunk(path, splitter)
        if chunks:
            print(f"   ✅ {path.name}: {len(chunks)} chunks")
            all_chunks.extend(chunks)

    if not all_chunks:
        raise SystemExit("No documents found to ingest.")

    print(f"\n📊 Total chunks: {len(all_chunks)}")

    # ── 2. ChromaDB vector ingestion ──
    print("\n--- 📦 VECTORS: Ingesting into ChromaDB ---")
    t0 = time.time()
    dict_chunks = to_vector_dicts(all_chunks)
    ingest_documents(dict_chunks, text_fields=["text"], recreate_collection=False)
    print(f"--- ✅ Vector ingestion done in {time.time() - t0:.1f}s ---")

    # ── 3. Neo4j graph ingestion (resume-safe) ──
    print("\n--- 🕸️ GRAPH: Ingesting into Neo4j AuraDB ---")
    t0 = time.time()
    ingest_into_graph(
        all_chunks,
        clear=False,  # Never clear on resume
        resume_from_checkpoint=True,
        checkpoint_path=checkpoint_path,
    )
    print(f"--- ✅ Graph ingestion done in {time.time() - t0:.1f}s ---")

    telemetry.log_info("Bulk ingestion completed", root=str(ROOT))
    print("\n✅ Ingestion complete! All chunks processed.")


if __name__ == "__main__":
    main()
