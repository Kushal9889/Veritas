"""
ACID-Compliant Checkpoint Manager for Veritas Ingestion Pipeline.

Uses SQLite WAL mode for crash-safe, atomic state persistence.
Tracks document hashes, per-chunk processing status, and extraction
statistics to enable instant resume after any failure.

Author: Veritas Engineering
"""
import os
import json
import sqlite3
import hashlib
import logging
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager

logger = logging.getLogger("ingestion.checkpoint")


class ChunkStatus(str, Enum):
    """Processing status for each chunk."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"  # No extractable entities


@dataclass
class ChunkRecord:
    """Immutable record of a processed chunk."""
    chunk_index: int
    status: ChunkStatus
    nodes_extracted: int = 0
    edges_extracted: int = 0
    model_used: str = ""
    error_message: str = ""
    processing_time_ms: float = 0.0


@dataclass
class IngestionState:
    """Complete ingestion pipeline state."""
    document_hash: str
    total_chunks: int
    processed_count: int = 0
    success_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    total_nodes: int = 0
    total_edges: int = 0
    last_processed_index: int = -1
    is_complete: bool = False


class CheckpointManager:
    """ACID-compliant checkpoint manager using SQLite WAL mode.
    
    Provides crash-safe state persistence for long-running ingestion
    pipelines. Tracks per-chunk processing status with atomic commits.
    
    Usage:
        mgr = CheckpointManager("data/ingestion/checkpoint.db")
        state = mgr.initialize_or_resume(doc_hash, total_chunks)
        
        for i in range(state.last_processed_index + 1, total_chunks):
            mgr.mark_in_progress(i)
            # ... process chunk ...
            mgr.mark_success(i, nodes=5, edges=3, model="qwen2.5:7b")
        
        mgr.finalize()
    """

    def __init__(self, db_path: str) -> None:
        """Initialize checkpoint manager.
        
        Args:
            db_path: Path to SQLite database file. Directory created if needed.
        """
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_schema()
        logger.info(f"CheckpointManager initialized: {db_path}")

    @contextmanager
    def _get_conn(self):
        """Context manager for database connections with WAL mode."""
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Create schema if not exists."""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS ingestion_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_hash TEXT NOT NULL,
                    total_chunks INTEGER NOT NULL,
                    processed_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    failed_count INTEGER DEFAULT 0,
                    skipped_count INTEGER DEFAULT 0,
                    total_nodes INTEGER DEFAULT 0,
                    total_edges INTEGER DEFAULT 0,
                    last_processed_index INTEGER DEFAULT -1,
                    is_complete INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chunk_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    nodes_extracted INTEGER DEFAULT 0,
                    edges_extracted INTEGER DEFAULT 0,
                    model_used TEXT DEFAULT '',
                    error_message TEXT DEFAULT '',
                    processing_time_ms REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES ingestion_runs(id),
                    UNIQUE(run_id, chunk_index)
                );

                CREATE INDEX IF NOT EXISTS idx_chunk_status
                    ON chunk_records(run_id, status);
                CREATE INDEX IF NOT EXISTS idx_run_hash
                    ON ingestion_runs(document_hash);
            """)

    def initialize_or_resume(
        self, document_hash: str, total_chunks: int
    ) -> IngestionState:
        """Initialize a new run or resume an existing incomplete run.
        
        Args:
            document_hash: SHA-256 hash of the source document.
            total_chunks: Total number of chunks to process.
            
        Returns:
            IngestionState with current progress.
        """
        with self._get_conn() as conn:
            # Check for existing incomplete run with same hash
            row = conn.execute(
                """SELECT * FROM ingestion_runs
                   WHERE document_hash = ? AND is_complete = 0
                   ORDER BY created_at DESC LIMIT 1""",
                (document_hash,)
            ).fetchone()

            if row:
                state = IngestionState(
                    document_hash=row["document_hash"],
                    total_chunks=row["total_chunks"],
                    processed_count=row["processed_count"],
                    success_count=row["success_count"],
                    failed_count=row["failed_count"],
                    skipped_count=row["skipped_count"],
                    total_nodes=row["total_nodes"],
                    total_edges=row["total_edges"],
                    last_processed_index=row["last_processed_index"],
                    is_complete=bool(row["is_complete"]),
                )
                self._current_run_id = row["id"]
                logger.info(
                    f"Resuming run #{self._current_run_id}: "
                    f"{state.processed_count}/{state.total_chunks} processed, "
                    f"resume from chunk {state.last_processed_index + 1}"
                )
                return state

            # Create new run
            cursor = conn.execute(
                """INSERT INTO ingestion_runs (document_hash, total_chunks)
                   VALUES (?, ?)""",
                (document_hash, total_chunks)
            )
            self._current_run_id = cursor.lastrowid
            logger.info(
                f"New ingestion run #{self._current_run_id}: "
                f"{total_chunks} chunks, hash={document_hash[:16]}..."
            )
            return IngestionState(
                document_hash=document_hash,
                total_chunks=total_chunks,
            )

    def mark_in_progress(self, chunk_index: int) -> None:
        """Mark a chunk as in-progress (atomic)."""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO chunk_records (run_id, chunk_index, status)
                   VALUES (?, ?, ?)
                   ON CONFLICT(run_id, chunk_index) DO UPDATE SET status = ?""",
                (self._current_run_id, chunk_index,
                 ChunkStatus.IN_PROGRESS, ChunkStatus.IN_PROGRESS)
            )

    def mark_success(
        self, chunk_index: int, nodes: int = 0, edges: int = 0,
        model: str = "", time_ms: float = 0.0
    ) -> None:
        """Mark a chunk as successfully processed (atomic)."""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO chunk_records
                       (run_id, chunk_index, status, nodes_extracted,
                        edges_extracted, model_used, processing_time_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(run_id, chunk_index) DO UPDATE SET
                       status=?, nodes_extracted=?, edges_extracted=?,
                       model_used=?, processing_time_ms=?""",
                (self._current_run_id, chunk_index, ChunkStatus.SUCCESS,
                 nodes, edges, model, time_ms,
                 ChunkStatus.SUCCESS, nodes, edges, model, time_ms)
            )
            conn.execute(
                """UPDATE ingestion_runs SET
                       processed_count = processed_count + 1,
                       success_count = success_count + 1,
                       total_nodes = total_nodes + ?,
                       total_edges = total_edges + ?,
                       last_processed_index = MAX(last_processed_index, ?),
                       updated_at = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (nodes, edges, chunk_index, self._current_run_id)
            )

    def mark_failed(
        self, chunk_index: int, error: str = "", model: str = ""
    ) -> None:
        """Mark a chunk as failed (atomic)."""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO chunk_records
                       (run_id, chunk_index, status, error_message, model_used)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(run_id, chunk_index) DO UPDATE SET
                       status=?, error_message=?, model_used=?""",
                (self._current_run_id, chunk_index, ChunkStatus.FAILED,
                 error, model, ChunkStatus.FAILED, error, model)
            )
            conn.execute(
                """UPDATE ingestion_runs SET
                       processed_count = processed_count + 1,
                       failed_count = failed_count + 1,
                       last_processed_index = MAX(last_processed_index, ?),
                       updated_at = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (chunk_index, self._current_run_id)
            )

    def mark_skipped(self, chunk_index: int) -> None:
        """Mark a chunk as skipped — no extractable entities."""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO chunk_records
                       (run_id, chunk_index, status)
                   VALUES (?, ?, ?)
                   ON CONFLICT(run_id, chunk_index) DO UPDATE SET status=?""",
                (self._current_run_id, chunk_index,
                 ChunkStatus.SKIPPED, ChunkStatus.SKIPPED)
            )
            conn.execute(
                """UPDATE ingestion_runs SET
                       processed_count = processed_count + 1,
                       skipped_count = skipped_count + 1,
                       last_processed_index = MAX(last_processed_index, ?),
                       updated_at = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (chunk_index, self._current_run_id)
            )

    def is_chunk_processed(self, chunk_index: int) -> bool:
        """Check if a specific chunk has already been processed."""
        with self._get_conn() as conn:
            row = conn.execute(
                """SELECT status FROM chunk_records
                   WHERE run_id = ? AND chunk_index = ?
                   AND status IN (?, ?)""",
                (self._current_run_id, chunk_index,
                 ChunkStatus.SUCCESS, ChunkStatus.SKIPPED)
            ).fetchone()
            return row is not None

    def get_state(self) -> IngestionState:
        """Get current ingestion state."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM ingestion_runs WHERE id = ?",
                (self._current_run_id,)
            ).fetchone()
            if not row:
                raise RuntimeError("No active ingestion run")
            return IngestionState(
                document_hash=row["document_hash"],
                total_chunks=row["total_chunks"],
                processed_count=row["processed_count"],
                success_count=row["success_count"],
                failed_count=row["failed_count"],
                skipped_count=row["skipped_count"],
                total_nodes=row["total_nodes"],
                total_edges=row["total_edges"],
                last_processed_index=row["last_processed_index"],
                is_complete=bool(row["is_complete"]),
            )

    def finalize(self) -> IngestionState:
        """Mark the current run as complete."""
        with self._get_conn() as conn:
            conn.execute(
                """UPDATE ingestion_runs SET is_complete = 1,
                       updated_at = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (self._current_run_id,)
            )
        state = self.get_state()
        logger.info(
            f"Ingestion run #{self._current_run_id} finalized: "
            f"{state.success_count} success, {state.failed_count} failed, "
            f"{state.skipped_count} skipped | "
            f"{state.total_nodes} nodes, {state.total_edges} edges"
        )
        return state

    def write_json_checkpoint(self, path: str) -> None:
        """Write a JSON checkpoint file for backward compatibility."""
        state = self.get_state()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "last_processed_index": state.last_processed_index,
            "total_chunks": state.total_chunks,
            "processed_count": state.processed_count,
            "success_count": state.success_count,
            "failed_count": state.failed_count,
            "total_nodes": state.total_nodes,
            "total_edges": state.total_edges,
        }
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp_path, path)  # Atomic rename


def compute_document_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file for deduplication.
    
    Args:
        file_path: Path to the file.
        
    Returns:
        Hex-encoded SHA-256 hash string.
    """
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha.update(block)
    return sha.hexdigest()


def compute_content_hash(texts: List[str]) -> str:
    """Compute SHA-256 hash of concatenated text content.
    
    Args:
        texts: List of text strings to hash.
        
    Returns:
        Hex-encoded SHA-256 hash string.
    """
    sha = hashlib.sha256()
    for t in texts:
        sha.update(t.encode("utf-8"))
    return sha.hexdigest()
