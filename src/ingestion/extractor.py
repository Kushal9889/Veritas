"""
Hybrid LLM Graph Extractor with Circuit Breaker Pattern.

Implements a fault-tolerant extraction pipeline that defaults to a primary
API (Gemini) and dynamically falls back to local Ollama models when rate
limits are hit. Uses the Circuit Breaker pattern with exponential backoff.

Author: Veritas Engineering
"""
import json
import time
import logging
import re
import requests
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from src.core.config import settings, GRAPH_EXTRACTION_PROMPT, NODE_LABELS, REL_TYPES
from src.core.telemetry import get_telemetry

logger = logging.getLogger("ingestion.extractor")
telemetry = get_telemetry("ingestion.extractor")


# ── Circuit Breaker ────────────────────────────────────────────────────

class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation — using primary API
    OPEN = "open"            # Primary API failed — using fallback
    HALF_OPEN = "half_open"  # Testing if primary API recovered


@dataclass
class CircuitBreaker:
    """Circuit Breaker for API rate-limit resilience.
    
    Tracks consecutive failures against the primary API. When the failure
    threshold is breached, the circuit opens and all traffic is routed to
    the local Ollama fallback. After a recovery timeout, the circuit
    enters half-open state and sends a single probe request to the primary.
    
    Attributes:
        failure_threshold: Consecutive failures before opening circuit.
        recovery_timeout_s: Seconds to wait before probing primary again.
    """
    failure_threshold: int = 3
    recovery_timeout_s: int = 120

    def __post_init__(self) -> None:
        self.state: CircuitState = CircuitState.CLOSED
        self.failure_count: int = 0
        self.last_failure_time: float = 0.0
        self.total_primary_calls: int = 0
        self.total_fallback_calls: int = 0

    def should_use_fallback(self) -> bool:
        """Determine whether to route to fallback."""
        if self.state == CircuitState.CLOSED:
            return False

        if self.state == CircuitState.OPEN:
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.recovery_timeout_s:
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit HALF_OPEN — probing primary API")
                return False  # Try primary once
            return True  # Still in cooldown

        # HALF_OPEN — let the single probe through
        return False

    def record_success(self) -> None:
        """Record a successful primary API call."""
        if self.state == CircuitState.HALF_OPEN:
            logger.info("Circuit CLOSED — primary API recovered ✅")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.total_primary_calls += 1

    def record_failure(self) -> None:
        """Record a primary API failure."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.total_fallback_calls += 1

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit OPEN — {self.failure_count} consecutive failures. "
                f"Routing to Ollama for {self.recovery_timeout_s}s."
            )
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning("Circuit re-OPENED — probe failed")


# ── JSON Parsing ───────────────────────────────────────────────────────

def _sanitize_json(raw: str) -> str:
    """Extract and clean JSON from LLM output.
    
    Handles markdown fences, leading text, and trailing garbage.
    
    Args:
        raw: Raw LLM output string.
        
    Returns:
        Cleaned JSON string ready for parsing.
    """
    if not raw:
        return ""

    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = cleaned.replace("```", "")

    # Find outermost JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start == -1 or end <= start:
        return ""
    return cleaned[start:end]


def _validate_graph_data(data: Dict[str, Any]) -> Dict[str, List[Dict]]:
    """Validate, normalize, and deduplicate extracted graph data.
    
    Enforces:
      - snake_case lowercase entity_id normalization for MERGE dedup
      - Schema-valid labels and relationship types
      - Edge endpoints must reference existing nodes
    
    Args:
        data: Raw parsed JSON from LLM.
        
    Returns:
        Validated dict with 'nodes' and 'edges' lists.
    """
    nodes: List[Dict] = []
    edges: List[Dict] = []

    valid_labels = set(NODE_LABELS)
    valid_rels = set(REL_TYPES)
    seen_ids: Dict[str, Dict] = {}  # normalized_id → node dict (dedup)

    def _normalize_entity_id(raw_id: str) -> str:
        """Normalize entity id to snake_case lowercase for deterministic MERGE."""
        # Strip punctuation, lowercase, collapse whitespace → underscores
        cleaned = re.sub(r"[^\w\s]", "", raw_id.lower().strip())
        cleaned = re.sub(r"\s+", "_", cleaned)
        # Remove trailing underscores
        return cleaned.strip("_")

    for node in data.get("nodes", []):
        raw_id = (node.get("id") or "").strip()
        n_type = (node.get("type") or "Entity").strip()
        n_name = (node.get("name") or raw_id).strip()
        if not raw_id:
            continue

        normalized_id = _normalize_entity_id(raw_id)
        if not normalized_id:
            continue

        # Normalize label — keep if valid, else default to Entity
        sanitized_type = "".join(c for c in n_type if c.isalnum())
        if sanitized_type not in valid_labels:
            sanitized_type = "Entity"

        # Deduplicate: first occurrence wins for type, keep best name
        if normalized_id not in seen_ids:
            seen_ids[normalized_id] = {
                "id": normalized_id, "type": sanitized_type, "name": n_name
            }

    nodes = list(seen_ids.values())
    node_ids = set(seen_ids.keys())

    for edge in data.get("edges", []):
        raw_src = (edge.get("source") or "").strip()
        raw_tgt = (edge.get("target") or "").strip()
        rel = (edge.get("type") or "RELATED_TO").strip()
        if not raw_src or not raw_tgt:
            continue

        src = _normalize_entity_id(raw_src)
        tgt = _normalize_entity_id(raw_tgt)

        # Normalize rel type
        sanitized_rel = "".join(
            c for c in rel if c.isalnum() or c == "_"
        ).upper()
        if sanitized_rel not in valid_rels:
            sanitized_rel = "RELATED_TO"

        # Only keep edges whose endpoints exist in nodes
        if src in node_ids and tgt in node_ids:
            edges.append({"source": src, "target": tgt, "type": sanitized_rel})

    return {"nodes": nodes, "edges": edges}


def parse_extraction_response(raw: str) -> Dict[str, List[Dict]]:
    """Parse and validate LLM extraction response.
    
    Args:
        raw: Raw LLM output string.
        
    Returns:
        Validated graph dict with 'nodes' and 'edges'.
    """
    cleaned = _sanitize_json(raw)
    if not cleaned:
        return {"nodes": [], "edges": []}
    try:
        data = json.loads(cleaned)
        return _validate_graph_data(data)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode failed: {e}")
        return {"nodes": [], "edges": []}


# ── LLM Backends ───────────────────────────────────────────────────────

def _call_ollama(prompt: str, model: str, timeout: int = 60) -> str:
    """Call Ollama local inference.
    
    Args:
        prompt: The extraction prompt.
        model: Ollama model name.
        timeout: Request timeout in seconds.
        
    Returns:
        Raw text response or empty string on failure.
    """
    try:
        res = requests.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 2048},
            },
            timeout=timeout,
        )
        if res.status_code != 200:
            logger.warning(f"Ollama returned {res.status_code} for model {model}")
            return ""
        return (res.json().get("response") or "").strip()
    except requests.exceptions.Timeout:
        logger.warning(f"Ollama timeout ({timeout}s) for model {model}")
        return ""
    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        return ""


def _call_gemini(prompt: str, model: str) -> str:
    """Call Gemini API with current API key.
    
    Args:
        prompt: The extraction prompt.
        model: Gemini model name.
        
    Returns:
        Raw text response.
        
    Raises:
        Exception on API error (including rate limits).
    """
    from google import genai
    client = genai.Client(api_key=settings.GOOGLE_API_KEY)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={"temperature": 0},
    )
    return (response.text or "").strip()


def _call_groq(prompt: str, model: str | None = None) -> str:
    """Call Groq API with round-robin key rotation across all available keys.
    
    Tracks per-key rate limits. When a key hits 429, marks it with a
    cooldown and rotates to the next available key. If ALL keys are
    exhausted, returns empty string to trigger Ollama fallback.
    
    Args:
        prompt: The extraction prompt.
        model: Override model name (defaults to settings.GROQ_MODEL).
        
    Returns:
        Raw text response or empty string on failure.
    """
    if not settings.GROQ_API_KEYS:
        return ""
    
    use_model = model or settings.GROQ_MODEL
    total_keys = len(settings.GROQ_API_KEYS)
    
    # Try each key once via round-robin
    for _ in range(total_keys):
        # Skip keys in cooldown
        if settings.all_groq_keys_exhausted():
            logger.warning("All Groq keys in cooldown")
            return ""
        
        current_idx = settings._current_groq_key_index
        try:
            from groq import Groq
            # max_retries=0 disables the SDK's built-in sleep-and-retry on 429s.
            # We handle rotation ourselves via round-robin key switching.
            client = Groq(api_key=settings.GROQ_API_KEY, max_retries=0)
            res = client.chat.completions.create(
                model=use_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2048,
            )
            return (res.choices[0].message.content or "").strip()
        except Exception as e:
            error_msg = str(e).lower()
            if "rate" in error_msg or "quota" in error_msg or "429" in error_msg:
                # Parse cooldown from error if available
                delay_match = re.search(r"try again in (\d+\.?\d*)", error_msg)
                cooldown = float(delay_match.group(1)) + 2 if delay_match else 60.0
                settings.mark_groq_key_cooldown(current_idx, cooldown)
                settings.get_next_groq_key()
                continue
            else:
                logger.warning(f"Groq non-rate-limit error (key #{current_idx+1}): {e}")
                settings.get_next_groq_key()
                continue
    return ""


def _is_rate_limit(e: Exception) -> bool:
    """Check if exception is a rate-limit / quota error."""
    msg = str(e).lower()
    return any(kw in msg for kw in (
        "429", "rate limit", "resource_exhausted", "quota", "too many requests"
    ))


def _parse_retry_delay(error_msg: str) -> int:
    """Extract retry delay from error message."""
    match = re.search(r"retry in (\d+)", error_msg, re.IGNORECASE)
    return int(match.group(1)) if match else 0


# ── Main Extractor ─────────────────────────────────────────────────────

# Module-level circuit breaker instance
_circuit = CircuitBreaker(
    failure_threshold=settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    recovery_timeout_s=settings.CIRCUIT_BREAKER_RECOVERY_TIMEOUT_S,
)


def extract_graph(text: str) -> Tuple[Dict[str, List[Dict]], str]:
    """Extract knowledge graph triples from text using cascading ModelRouter.
    
    Execution order (ingestion only — never affects retrieval/generation):
      1. Groq primary model (9-key round-robin, per-key cooldown tracking)
      2. Groq fallback model (lighter model, same 9-key rotation)
      3. Gemini (4-key rotation + circuit breaker)
      4. Ollama primary model (qwen2.5:7b — unlimited local)
      5. Ollama fallback model (phi3.5:3.8b — last resort)
    
    Args:
        text: Raw text chunk to extract entities/relationships from.
        
    Returns:
        Tuple of (graph_data_dict, model_name_used).
        graph_data_dict has keys 'nodes' and 'edges'.
    """
    prompt = GRAPH_EXTRACTION_PROMPT.format(text=text)
    empty = ({"nodes": [], "edges": []}, "none")

    # ── Strategy 1: Groq primary model (9-key round-robin) ─────────
    if settings.GROQ_API_KEYS and not settings.all_groq_keys_exhausted():
        raw = _call_groq(prompt, settings.GROQ_MODEL)
        if raw:
            result = parse_extraction_response(raw)
            if result["nodes"]:
                return result, f"groq/{settings.GROQ_MODEL}"

    # ── Strategy 2: Groq fallback model (lighter, same keys) ───────
    if (settings.GROQ_API_KEYS
            and settings.GROQ_FALLBACK_MODEL != settings.GROQ_MODEL
            and not settings.all_groq_keys_exhausted()):
        logger.info(f"Groq primary exhausted/failed → trying fallback model: {settings.GROQ_FALLBACK_MODEL}")
        raw = _call_groq(prompt, settings.GROQ_FALLBACK_MODEL)
        if raw:
            result = parse_extraction_response(raw)
            if result["nodes"]:
                return result, f"groq/{settings.GROQ_FALLBACK_MODEL}"

    # ── Strategy 3: Gemini (4-key rotation + circuit breaker) ──────
    if settings.GOOGLE_API_KEYS and not _circuit.should_use_fallback():
        for attempt in range(len(settings.GOOGLE_API_KEYS)):
            try:
                if settings.GRAPH_EXTRACTION_THROTTLE_S > 0:
                    time.sleep(settings.GRAPH_EXTRACTION_THROTTLE_S)

                raw = _call_gemini(prompt, settings.GRAPH_EXTRACTION_MODEL)
                result = parse_extraction_response(raw)
                _circuit.record_success()
                return result, f"gemini/{settings.GRAPH_EXTRACTION_MODEL}"

            except Exception as e:
                if _is_rate_limit(e):
                    _circuit.record_failure()
                    settings.get_next_api_key()
                    delay = _parse_retry_delay(str(e))
                    wait = max(delay + 1, 2)
                    logger.warning(
                        f"Gemini rate limit (key #{settings._current_gemini_key_index + 1}). "
                        f"Wait {wait}s. Circuit: {_circuit.state.value}"
                    )
                    time.sleep(wait)
                else:
                    _circuit.record_failure()
                    logger.error(f"Gemini non-rate-limit error: {e}")
                    break

    # ── Strategy 4: Ollama primary model ───────────────────────────
    primary_model = settings.GRAPH_EXTRACTION_OLLAMA_MODEL
    logger.debug(f"All APIs exhausted → Ollama primary: {primary_model}")
    raw = _call_ollama(prompt, primary_model)
    if raw:
        result = parse_extraction_response(raw)
        if result["nodes"]:
            return result, f"ollama/{primary_model}"

    # ── Strategy 5: Ollama fallback model ──────────────────────────
    fallback_model = settings.GRAPH_EXTRACTION_OLLAMA_FALLBACK
    if fallback_model != primary_model:
        logger.info(f"Ollama primary failed → fallback: {fallback_model}")
        raw = _call_ollama(prompt, fallback_model)
        if raw:
            result = parse_extraction_response(raw)
            if result["nodes"]:
                return result, f"ollama/{fallback_model}"

    # ── All 5 strategies exhausted ─────────────────────────────────
    logger.warning("All extraction strategies exhausted for chunk")
    return empty
