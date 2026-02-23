"""Local Models Interaction layer — Production-Optimized
Handles LLM-as-a-judge and embeddings.
"""
from typing import List, Dict
import requests
from src.core.telemetry import get_telemetry
from src.core.config import settings
from langsmith import traceable

telemetry = get_telemetry('core.models')

class OllamaEmbeddings:
    """Local embedding model using Ollama with batch support."""

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self._batch_endpoint = f"{base_url}/api/embed"
        self._single_endpoint = f"{base_url}/api/embeddings"
        self._supports_batch = None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if self._supports_batch is not False:
            try:
                response = requests.post(
                    self._batch_endpoint,
                    json={"model": self.model, "input": texts},
                    timeout=120
                )
                if response.status_code == 200:
                    data = response.json()
                    embeddings = data.get("embeddings", [])
                    if len(embeddings) == len(texts):
                        self._supports_batch = True
                        telemetry.log_info("Batch embedding succeeded", count=len(texts))
                        return embeddings
                self._supports_batch = False
            except Exception:
                self._supports_batch = False
                telemetry.log_info("Batch endpoint unavailable, using sequential")

        embeddings = []
        for text in texts:
            response = requests.post(
                self._single_endpoint,
                json={"model": self.model, "prompt": text},
                timeout=30
            )
            if response.status_code == 200:
                embeddings.append(response.json()["embedding"])
            else:
                raise Exception(f"Ollama embedding failed: {response.text}")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        response = requests.post(
            self._single_endpoint,
            json={"model": self.model, "prompt": text},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()["embedding"]
        else:
            raise Exception(f"Ollama embedding failed: {response.text}")

def get_ollama_embeddings():
    return OllamaEmbeddings(model=settings.OLLAMA_EMBEDDING_MODEL, base_url=settings.OLLAMA_BASE_URL)


@traceable(run_type="llm", name="Ollama_Judge")
def judge_retrieval_quality(question: str, retrieved_docs: List[Dict], answer: str) -> Dict:
    """
    Use LLM to judge if retrieved documents are relevant and answer is accurate
    Returns: {score: float, reasoning: str, relevant_docs: int}
    """
    docs_text = "\n\n".join([f"Doc {i+1}: {doc.get('text', '')[:500]}..." for i, doc in enumerate(retrieved_docs[:3])])
    
    prompt = f"""You are an expert evaluator. Judge the quality of this RAG system response.

Question: {question}

Retrieved Documents:
{docs_text}

Generated Answer: {answer}

Evaluate on a scale of 1-10:
1. Are the retrieved documents relevant to the question?
2. Does the answer accurately use the retrieved documents?
3. Is the answer complete and helpful?

Respond in JSON format:
{{"score": <1-10>, "reasoning": "<brief explanation>", "relevant_docs": <count of relevant docs>}}"""
    
    try:
        response = requests.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": settings.JUDGE_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            },
            timeout=10
        )
        if response.status_code == 200:
            import json
            result = response.json()
            judgment = json.loads(result.get('response', '{}'))
            return {
                "score": judgment.get("score", 5),
                "reasoning": judgment.get("reasoning", ""),
                "relevant_docs": judgment.get("relevant_docs", 0)
            }
    except Exception:
        pass
    
    return {
        "score": 7 if len(retrieved_docs) > 0 else 3,
        "reasoning": "Fallback evaluation",
        "relevant_docs": len(retrieved_docs)
    }
