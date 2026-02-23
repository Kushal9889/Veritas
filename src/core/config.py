import os
import time
from dotenv import load_dotenv
from typing import List, Dict, Any

# loading envs from .env file
load_dotenv()

# ── Neo4j Graph Schema Constants ───────────────────────────────────────
# Canonical node labels for 10-K knowledge graph (15+ entity types)
NODE_LABELS = (
    "Company", "Subsidiary", "Executive", "Director", "Person",
    "Filing", "FinancialMetric", "RiskFactor", "Product", "Service",
    "Location", "Industry", "Segment", "Technology", "Regulation",
    "LegalProceeding", "Contract", "Patent", "Organization",
    "Competitor", "Supplier", "Customer", "Entity", "Chunk",
)

# Canonical relationship types
REL_TYPES = (
    "MENTIONS_RISK", "IMPACTS_METRIC", "PRODUCES", "OFFERS_SERVICE",
    "OWNS", "HAS_SUBSIDIARY", "COMPETES_WITH", "LOCATED_IN",
    "REPORTS_METRIC", "EXPOSED_TO_RISK", "FILED_BY", "RELATED_TO",
    "OPERATES_IN", "EMPLOYS", "REGULATES", "SUPPLIES_TO",
    "CUSTOMER_OF", "HOLDS_PATENT", "PARTY_TO_CONTRACT",
    "SUBJECT_TO_LITIGATION", "SERVES_MARKET", "LED_BY",
)

# ── Extraction Prompt (strict JSON schema with entity normalization) ────
GRAPH_EXTRACTION_PROMPT = """You are an elite financial knowledge graph engineer analyzing SEC 10-K filings.
Extract entities and relationships from the text below. Return ONLY valid JSON. No markdown fences. No text outside JSON.

STRICT OUTPUT SCHEMA:
{{
  "nodes": [
    {{"id": "<normalized_entity_id>", "type": "<EntityType>", "name": "<display_name>"}}
  ],
  "edges": [
    {{"source": "<source_id>", "target": "<target_id>", "type": "<RelationType>"}}
  ]
}}

ENTITY TYPES (use EXACTLY these):
Company, Subsidiary, Executive, Director, Person, Filing, FinancialMetric,
RiskFactor, Product, Service, Location, Industry, Segment, Technology,
Regulation, LegalProceeding, Contract, Patent, Organization, Competitor,
Supplier, Customer

RELATIONSHIP TYPES (use EXACTLY these):
MENTIONS_RISK, IMPACTS_METRIC, PRODUCES, OFFERS_SERVICE, OWNS,
HAS_SUBSIDIARY, COMPETES_WITH, LOCATED_IN, REPORTS_METRIC,
EXPOSED_TO_RISK, FILED_BY, RELATED_TO, OPERATES_IN, EMPLOYS,
REGULATES, SUPPLIES_TO, CUSTOMER_OF, HOLDS_PATENT,
PARTY_TO_CONTRACT, SUBJECT_TO_LITIGATION, SERVES_MARKET, LED_BY

CRITICAL ENTITY NORMALIZATION RULES:
1. The "id" field MUST be a snake_case, lowercase, canonical identifier.
   - "Apple Inc.", "Apple", "Apple Corporation", "AAPL" → id: "apple_inc"
   - "Elon Musk", "Mr. Musk", "CEO Elon Musk" → id: "elon_musk"
   - "Revenue", "Total Revenue", "Net Revenue" → id: "total_revenue"
   - "SEC", "Securities and Exchange Commission" → id: "sec"
2. The "name" field stores the human-readable display form.
3. Every edge source/target MUST reference an id from the nodes list.
4. Minimum 1 node per chunk. If no entities found, return {{"nodes":[],"edges":[]}}.
5. Extract ONLY what is stated. Do NOT invent data.
6. Be aggressive about deduplication: same real-world entity = same id.

TEXT:
{text}"""


class Settings:
    def __init__(self):
        # google api keys for gemini with fallback support
        self.GOOGLE_API_KEYS: List[str] = [
            key for key in [
                os.getenv("GOOGLE_API_KEY"),
                os.getenv("GOOGLE_API_KEY_2"),
                os.getenv("GOOGLE_API_KEY_3"),
                os.getenv("GOOGLE_API_KEY_4")
            ] if key
        ]
        self.GOOGLE_API_KEY: str | None = self.GOOGLE_API_KEYS[0] if self.GOOGLE_API_KEYS else None
        self._current_gemini_key_index: int = 0
        
        # groq api keys for high-priority ingestion (up to 9 keys for round-robin)
        self.GROQ_API_KEYS: List[str] = [
            key for key in [
                os.getenv(f"GROQ_API_KEY_{i}") for i in range(1, 10)
            ] if key
        ]
        self.GROQ_API_KEY: str | None = self.GROQ_API_KEYS[0] if self.GROQ_API_KEYS else None
        self._current_groq_key_index: int = 0
        self.GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.GROQ_FALLBACK_MODEL: str = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")
        # Per-key rate-limit tracking: key_index → epoch when usable again
        self._groq_key_cooldowns: Dict[int, float] = {}
        
        # chroma config
        self.CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "veritas-financial-index")
        self.CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        
        # neo4j graph database config
        raw_uri = os.getenv("NEO4J_URI", "")
        if ":7687" in raw_uri:
            raw_uri = raw_uri.replace(":7687", "")
        self.NEO4J_URI: str = raw_uri.strip()
        self.NEO4J_USERNAME: str | None = os.getenv("NEO4J_USERNAME")
        self.NEO4J_PASSWORD: str | None = os.getenv("NEO4J_PASSWORD")
        
        # langsmith config for tracing
        self.LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        self.LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", os.getenv("LANGSMITH_API_KEY", ""))
        self.LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "veritas-graph-rag")
        
        # embedding provider config
        self.EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "ollama")
        self.OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        self.OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # LLM models config
        self.ROUTER_MODEL: str = os.getenv("ROUTER_MODEL", "phi3.5:3.8b")
        self.GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "models/gemini-2.5-flash")
        self.JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "phi3.5:3.8b")
        
        # Graph extraction — Ollama models (primary + fallback)
        self.GRAPH_EXTRACTION_OLLAMA_MODEL: str = os.getenv("GRAPH_EXTRACTION_OLLAMA_MODEL", "qwen2.5:7b")
        self.GRAPH_EXTRACTION_OLLAMA_FALLBACK: str = os.getenv("GRAPH_EXTRACTION_OLLAMA_FALLBACK", "phi3.5:3.8b")

        # Graph extraction — Gemini/API config
        self.GRAPH_EXTRACTION_PROVIDER: str = os.getenv("GRAPH_EXTRACTION_PROVIDER", "gemini").lower().strip()
        self.GRAPH_EXTRACTION_MODEL: str = os.getenv("GRAPH_EXTRACTION_MODEL", "models/gemini-2.5-flash")
        self.GRAPH_EXTRACTION_MAX_RETRIES: int = int(os.getenv("GRAPH_EXTRACTION_MAX_RETRIES", "8"))
        self.GRAPH_EXTRACTION_MAX_WAIT_S: int = int(os.getenv("GRAPH_EXTRACTION_MAX_WAIT_S", "120"))
        self.GRAPH_EXTRACTION_THROTTLE_S: float = float(os.getenv("GRAPH_EXTRACTION_THROTTLE_S", "0"))

        # Circuit Breaker config
        self.CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "3"))
        self.CIRCUIT_BREAKER_RECOVERY_TIMEOUT_S: int = int(os.getenv("CIRCUIT_BREAKER_RECOVERY_TIMEOUT_S", "120"))

        # Checkpoint config
        self.CHECKPOINT_DB_PATH: str = os.getenv("CHECKPOINT_DB_PATH", "data/ingestion/checkpoint.db")
        self.CHECKPOINT_JSON_PATH: str = os.getenv("GRAPH_CHECKPOINT", "data/ingestion/graph_checkpoint.json")

        # Neo4j batch config
        self.NEO4J_BATCH_SIZE: int = int(os.getenv("NEO4J_BATCH_SIZE", "50"))
        
        self._validate()
        self._setup_langsmith()

    def _validate(self) -> None:
        missing: List[str] = []
        
        if not self.GOOGLE_API_KEYS:
            missing.append("GOOGLE_API_KEY")
            
        if not self.NEO4J_URI:
            missing.append("NEO4J_URI")
        if not self.NEO4J_USERNAME:
            missing.append("NEO4J_USERNAME")
        if not self.NEO4J_PASSWORD:
            missing.append("NEO4J_PASSWORD")
        
        if missing:
            raise ValueError(f"critical config error: missing environment variables: {', '.join(missing)}")
    
    def _setup_langsmith(self) -> None:
        """Setup LangSmith tracing if enabled."""
        if self.LANGCHAIN_TRACING_V2 and self.LANGCHAIN_API_KEY:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.LANGCHAIN_API_KEY
            os.environ["LANGCHAIN_PROJECT"] = self.LANGCHAIN_PROJECT
            print(f"✅ LangSmith tracing enabled - Project: {self.LANGCHAIN_PROJECT}")
        else:
            print("ℹ️  LangSmith tracing disabled (set LANGCHAIN_TRACING_V2=true to enable)")
    
    def get_next_api_key(self) -> str | None:
        """Rotate to next available Gemini API key on rate limit."""
        if len(self.GOOGLE_API_KEYS) <= 1:
            return self.GOOGLE_API_KEY
        self._current_gemini_key_index = (self._current_gemini_key_index + 1) % len(self.GOOGLE_API_KEYS)
        self.GOOGLE_API_KEY = self.GOOGLE_API_KEYS[self._current_gemini_key_index]
        print(f"🔄 Switched to Gemini API key #{self._current_gemini_key_index + 1}")
        return self.GOOGLE_API_KEY
    
    def get_next_groq_key(self) -> str | None:
        """Round-robin to next Groq API key (no cooldown/wait)."""
        if not self.GROQ_API_KEYS:
            return None
        total = len(self.GROQ_API_KEYS)
        # simple round-robin rotation without cooldown checks
        self._current_groq_key_index = (self._current_groq_key_index + 1) % total
        self.GROQ_API_KEY = self.GROQ_API_KEYS[self._current_groq_key_index]
        print(f"🔄 Switched to Groq key #{self._current_groq_key_index + 1}/{total}")
        return self.GROQ_API_KEY

    def mark_groq_key_cooldown(self, key_index: int, cooldown_seconds: float = 0.0) -> None:
        """No-op: cooldowns are disabled. Kept for backward compatibility."""
        # Intentionally do nothing so rotation is purely round-robin.
        print(f"   ⚪ Groq key cooldowns disabled (would have cooled key #{key_index + 1})")

    def get_available_groq_key_count(self) -> int:
        """Count how many Groq keys are currently usable (not in cooldown)."""
        # cooldowns disabled -> all keys considered available
        return len(self.GROQ_API_KEYS)

    def all_groq_keys_exhausted(self) -> bool:
        """Check if ALL Groq keys are currently in cooldown."""
        # cooldowns disabled -> never consider all keys exhausted
        return False

# initializing the settings object
settings = Settings()

def get_gemini_client_with_fallback():
    """Get Gemini client with automatic API key rotation on rate limits."""
    from google import genai
    return genai.Client(api_key=settings.GOOGLE_API_KEY)
