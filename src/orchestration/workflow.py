"""
Unified Orchestration Workflow
Handles LangGraph State Machine, Routing, Retrieval, and Generation.
"""
import json
import time
import requests
import threading
import xml.sax.saxutils
from typing import List, TypedDict

from google import genai
from google.genai import types
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langgraph.graph import StateGraph, END
from langsmith import traceable

from src.core.config import settings, get_gemini_client_with_fallback
from src.core.telemetry import get_telemetry
from src.retrieval.search import query_financial_docs, query_graph

telemetry = get_telemetry('orchestration.workflow')

# ── 1. Agent State ─────────────────────────────────────────────────────
class AgentState(TypedDict):
    question: str
    query_category: str
    metadata_filter: dict
    entities: List[str]
    router_keywords: List[str]
    documents: List[str]
    graph_data: List[str]
    generation: str
    is_grounded: bool
    critic_feedback: List[str]
    loop_count: int

# ── 2. Router Logic ────────────────────────────────────────────────────
CATEGORY_KEYWORDS = {
    "financial": ["revenue", "profit", "income", "earnings", "cash flow", "balance sheet",
                  "financial", "margin", "expense", "cost", "sales", "debt", "assets"],
    "risk": ["risk", "danger", "threat", "regulatory", "uncertainty", "litigation",
             "compliance", "volatility", "competition", "cybersecurity"],
    "operations": ["business", "operations", "product", "manufacturing", "strategy",
                   "segment", "supply chain", "factory", "production", "vehicle", "energy"],
    "legal": ["legal", "lawsuit", "litigation", "court", "compliance",
              "governance", "regulation", "SEC", "filing"],
    "general": ["hello", "hi", "hey", "thanks", "bye", "who are you"],
}

ROUTER_PROMPT = """You are a query classifier for a financial document Q&A system.
Classify the user's question into exactly ONE category, extract relevant tags, and extract entity names.

Categories:
- financial: Questions about revenue, profit, earnings, cash flow, financial performance
- risk: Questions about risk factors, threats, regulatory issues, uncertainties
- operations: Questions about business operations, products, strategy, manufacturing
- legal: Questions about legal proceedings, compliance, governance
- general: Greetings, small talk, or questions unrelated to financial documents

Respond with ONLY valid JSON, no other text:
{{"category": "<category>", "tags": ["<tag1>", "<tag2>"], "entities": ["<company_or_person>"]}}

Tags: 1-3 specific topic keywords from the question.
Entities: Company names, people, or products mentioned. If none, set to [].

Question: "{question}"
"""

def _keyword_classify(question: str) -> dict:
    q_lower = question.lower().strip()
    greetings = ["hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye"]
    if q_lower in greetings or len(question.split()) < 3:
        return {"category": "general", "tags": [], "entities": []}
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        if category == "general":
            continue
        score = sum(1 for kw in keywords if kw in q_lower)
        if score > 0:
            scores[category] = score
    if scores:
        best = max(scores, key=scores.get)
        matched = [kw for kw in CATEGORY_KEYWORDS[best] if kw in q_lower]
        return {"category": best, "tags": matched[:3], "entities": []}
    return {"category": "financial", "tags": [], "entities": []}

def classify_query(question: str) -> dict:
    start_time = time.time()
    try:
        response = requests.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": settings.ROUTER_MODEL,
                "prompt": ROUTER_PROMPT.format(question=question),
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 120}
            },
            timeout=5
        )
        if response.status_code == 200:
            raw = response.json().get("response", "").strip()
            json_start = raw.find("{")
            json_end = raw.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                parsed = json.loads(raw[json_start:json_end])
                category = parsed.get("category", "").lower().strip()
                tags = parsed.get("tags", [])
                entities = parsed.get("entities", [])
                if category in CATEGORY_KEYWORDS:
                    duration = time.time() - start_time
                    telemetry.log_info("SML routing succeeded", category=category, tags=tags, entities=entities, duration_ms=round(duration * 1000, 2))
                    telemetry.track_metric("router_duration", duration)
                    return {"category": category, "tags": tags, "entities": entities}
    except Exception as e:
        telemetry.log_warning("SML routing failed, falling back to keyword router", error=str(e))
    
    result = _keyword_classify(question)
    duration = time.time() - start_time
    telemetry.log_info("Keyword routing used (fallback)", category=result["category"], duration_ms=round(duration * 1000, 2))
    return result

def build_metadata_filter(category: str) -> dict:
    if category == "general":
        return None
    return {"section": category}

def route_node(state: AgentState):
    question = state["question"]
    telemetry.log_info("🔀 Router processing query", question=question[:100])
    classification = classify_query(question)
    category = classification["category"]
    tags = classification["tags"]
    entities = classification["entities"]
    metadata_filter = build_metadata_filter(category)
    telemetry.log_info(f"🏷️ Routed to: {category}", tags=tags, entities=entities, has_filter=metadata_filter is not None)
    print(f"  🔀 Router: category={category}, tags={tags}, entities={entities}")
    return {"query_category": category, "metadata_filter": metadata_filter or {}, "entities": entities, "router_keywords": tags}

# ── 3. Node Logic (Retrieval & Generation) ─────────────────────────────
_gemini_client = None
def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = get_gemini_client_with_fallback()
    return _gemini_client

def _refresh_gemini_client():
    global _gemini_client
    _gemini_client = get_gemini_client_with_fallback()
    return _gemini_client

CATEGORY_PERSONA = {
    "financial": "investment banker",
    "risk": "chief risk officer",
    "operations": "senior business analyst",
    "legal": "forensic accountant",
    "general": "financial assistant",
}

def get_persona(category: str) -> str:
    return CATEGORY_PERSONA.get(category, "senior financial analyst")

def _async_judge(question, documents, full_text):
    try:
        from src.core.models import judge_retrieval_quality
        judgment = judge_retrieval_quality(question, [{"text": d} for d in documents], full_text)
        telemetry.track_metric("judge_score", judgment["score"])
    except Exception as e:
        telemetry.log_warning("Async judge failed", error=str(e))

def _run_graph_queries(entities: list) -> list:
    graph_texts = []
    if not entities:
        return graph_texts
    try:
        clean_entities = [e.strip() for e in entities if e and e.strip()]
        if not clean_entities:
            return graph_texts
        if len(clean_entities) == 1:
            connections = query_graph(clean_entities[0])
            if connections:
                for c in connections[:5]:
                    graph_texts.append(f"{c['source']} → {c['relationship']} → {c['target']}")
        else:
            graph_queries = {}
            for i, entity in enumerate(clean_entities[:3]):
                graph_queries[f"e{i}"] = RunnableLambda(lambda _input, e=entity: query_graph(e)).with_config({"run_name": f"GraphQuery_{entity}"})
            results = RunnableParallel(**graph_queries).with_config({"run_name": "ParallelGraphQueries"}).invoke({})
            for connections in results.values():
                if connections:
                    for c in connections[:5]:
                        graph_texts.append(f"{c['source']} → {c['relationship']} → {c['target']}")
    except Exception as e:
        telemetry.log_warning("Graph search skipped", error=str(e))
    return graph_texts

def retrieve_node(state: AgentState):
    start_time = time.time()
    question = state["question"]
    metadata_filter = state.get("metadata_filter", None)
    query_category = state.get("query_category", "unknown")
    entities = state.get("entities", [])
    router_keywords = state.get("router_keywords", [])

    # ── Entity fallback: if the LLM router returned no entities (e.g. keyword
    # fallback was used due to Ollama timeout), extract candidate entity strings
    # directly from the question so graph retrieval is never silently skipped.
    if not entities:
        entities = [
            w.rstrip(".,?!;:")
            for w in question.split()
            if len(w) > 2 and w[0].isupper() and w.isalpha()
        ]
        if entities:
            telemetry.log_info("Entity fallback: extracted from question", entities=entities)

    telemetry.log_info("Starting hybrid retrieval", question=question[:100], category=query_category)

    vector_search = RunnableLambda(
        lambda q: query_financial_docs(q, top_k=5, metadata_filter=metadata_filter if metadata_filter else None, router_keywords=router_keywords)
    ).with_config({"run_name": "VectorSearch"})

    graph_search = RunnableLambda(lambda q: _run_graph_queries(entities)).with_config({"run_name": "GraphSearch"})

    parallel_chain = RunnableParallel(vector_results=vector_search, graph_texts=graph_search).with_config({"run_name": "ParallelHybridRetrieval"})
    result = parallel_chain.invoke(question)
    
    doc_texts = [f"[{r['source']} p{r['page']}] {r['text']}" for r in result["vector_results"]]
    total_duration = time.time() - start_time
    telemetry.log_info("Hybrid retrieval completed", total_duration_ms=round(total_duration * 1000, 2), vector_results=len(doc_texts), graph_results=len(result["graph_texts"]))
    return {"documents": doc_texts, "graph_data": result["graph_texts"]}

SYSTEM_PROMPT = "You are a {role} analyzing SEC 10-K filings.\nRules:\n- Answer using ONLY the evidence below\n- Cite sources as [Source, pN]\n- If evidence is insufficient, say so\n- Be concise and precise"

@traceable(run_type="llm", name="Gemini_Generation")
def _invoke_gemini(prompt: str, model_name: str) -> str:
    client = _get_gemini_client()
    for attempt in range(len(settings.GOOGLE_API_KEYS)):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(max_output_tokens=2000, temperature=0.3)
            )
            return response.text
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                settings.get_next_api_key()
                client = _refresh_gemini_client()
                time.sleep(2)
            else:
                raise
    raise Exception("All Gemini API keys exhausted")

@traceable(run_type="llm", name="Groq_Generation")
def _invoke_groq(prompt: str, model_name: str) -> str:
    from groq import Groq
    # max_retries=0: disable SDK sleep-and-retry so 429s surface immediately
    # and our own key-rotation logic (in extractor) handles fallback.
    client = Groq(api_key=settings.GROQ_API_KEY, max_retries=0)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=2000
    )
    return response.choices[0].message.content

def generate_node(state: AgentState, stream_callback=None):
    start_time = time.time()
    question = state["question"]
    documents = state["documents"]
    graph_data = state.get("graph_data", [])
    query_category = state.get("query_category", "")

    if not documents and not graph_data:
        if query_category == "general":
            return {"generation": "Hello! I'm Veritas, your financial document assistant. Ask me anything about Tesla's 10-K filing — revenue, risks, operations, or legal matters."}
        return {"generation": "I couldn't find relevant information. Try asking about specific financial metrics, risk factors, or business operations from the 10-K filing."}

    xml_evidence = "<context>\n"
    if documents:
        xml_evidence += "  <vector_retrieval>\n"
        for i, doc in enumerate(documents, 1):
            escaped_doc = xml.sax.saxutils.escape(doc)
            xml_evidence += f"    <document id=\"{i}\">\n      {escaped_doc}\n    </document>\n"
        xml_evidence += "  </vector_retrieval>\n"
    
    if graph_data:
        xml_evidence += "  <knowledge_graph>\n"
        for g in graph_data:
            escaped_g = xml.sax.saxutils.escape(g)
            xml_evidence += f"    <relationship>{escaped_g}</relationship>\n"
        xml_evidence += "  </knowledge_graph>\n"
    
    xml_evidence += "</context>"

    critic_feedback = state.get("critic_feedback", [])
    if critic_feedback:
        escaped_feedback = [xml.sax.saxutils.escape(f) for f in critic_feedback]
        xml_evidence += "\n\n<critic_feedback>\n"
        xml_evidence += "The previous draft was rejected for hallucinations. You MUST fix the following:\n"
        for f in escaped_feedback:
            xml_evidence += f"  <issue>{f}</issue>\n"
        xml_evidence += "</critic_feedback>\n"

    prompt = f"{SYSTEM_PROMPT.format(role=get_persona(query_category))}\n\n{xml_evidence}\n\n<question>\n{question}\n</question>"

    if settings.GENERATION_MODEL.startswith("models/") or "gemini" in settings.GENERATION_MODEL:
        try:
            full_text = _invoke_gemini(prompt, settings.GENERATION_MODEL)
            if stream_callback:
                stream_callback(full_text)
            else:
                print(full_text, end='', flush=True)

            duration = time.time() - start_time
            telemetry.log_info("Generation completed", duration_ms=round(duration * 1000, 2))
            telemetry.track_metric("generation_duration", duration)

            threading.Thread(target=_async_judge, args=(question, documents, full_text), daemon=True).start()
            new_loop = (state.get("loop_count") or 0) + 1
            return {"generation": full_text, "loop_count": new_loop}
        except Exception as e:
            telemetry.log_error("Gemini generation failed", error=str(e))

    try:
        response = requests.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={"model": settings.GENERATION_MODEL, "prompt": prompt, "stream": True},
            stream=True, timeout=60
        )
        if response.status_code == 200:
            import json as json_mod
            full_text = ""
            for line in response.iter_lines():
                if line:
                    chunk = json_mod.loads(line)
                    if 'response' in chunk:
                        text = chunk['response']
                        full_text += text
                        if stream_callback:
                            stream_callback(text)
                        else:
                            print(text, end='', flush=True)
            telemetry.log_info("Ollama generation successful")
            new_loop = (state.get("loop_count") or 0) + 1
            return {"generation": full_text, "loop_count": new_loop}
    except Exception as e:
        telemetry.log_error("Generation failed", error=str(e))
    return {"generation": "Error: Generation failed. Please try again.", "loop_count": (state.get("loop_count") or 0) + 1}

# ── 4. Critic Node ─────────────────────────────────────────────────────
CRITIC_PROMPT = """You are an adversarial SEC Auditor. You are given a <Draft> and the verified <Evidence>. 
Your ONLY job is to verify that EVERY metric, date, and claim in the <Draft> is explicitly stated in the <Evidence>. 
If a claim is not in the <Evidence>, you must output a JSON object with is_grounded: false and hallucinated_claims.
If all claims are backed by evidence, output {{"is_grounded": true}}.
Output strictly valid JSON.

Evidence:
{evidence}

Draft to Critique:
{draft}
"""

def critic_node(state: AgentState):
    start_time = time.time()
    telemetry.log_info("Critic auditing generation for hallucinations")
    
    draft = state.get("generation", "")
    documents = state.get("documents", [])
    graph_data = state.get("graph_data", [])
    
    xml_evidence = "<context>\n"
    if documents:
        xml_evidence += "  <vector_retrieval>\n"
        for i, doc in enumerate(documents, 1):
            escaped_doc = xml.sax.saxutils.escape(doc)
            xml_evidence += f"    <document id=\"{i}\">\n      {escaped_doc}\n    </document>\n"
        xml_evidence += "  </vector_retrieval>\n"
    if graph_data:
        xml_evidence += "  <knowledge_graph>\n"
        for g in graph_data:
            escaped_g = xml.sax.saxutils.escape(g)
            xml_evidence += f"    <relationship>{escaped_g}</relationship>\n"
        xml_evidence += "  </knowledge_graph>\n"
    xml_evidence += "</context>"

    prompt = CRITIC_PROMPT.format(evidence=xml_evidence, draft=draft)
    
    is_grounded = True
    feedback = []
    
    try:
        # Use generated model for audit
        if settings.GENERATION_MODEL.startswith("models/") or "gemini" in settings.GENERATION_MODEL:
            client = _get_gemini_client()
            for attempt in range(len(settings.GOOGLE_API_KEYS)):
                try:
                    res = client.models.generate_content(
                        model=settings.GENERATION_MODEL,
                        contents=prompt,
                        config=types.GenerateContentConfig(temperature=0.0)
                    )
                    raw = res.text.strip()
                    break
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower():
                        settings.get_next_api_key()
                        client = _refresh_gemini_client()
                        time.sleep(2)
                    else:
                        raw = '{"is_grounded": true}'
                        break
        else:
            res = requests.post(f"{settings.OLLAMA_BASE_URL}/api/generate", json={
                "model": settings.ROUTER_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0}
            }, timeout=10)
            if res.status_code == 200:
                raw = res.json().get("response", "").strip()
            else:
                raw = '{"is_grounded": true}'
        
        # Safely parse JSON from raw response
        json_start = raw.find("{")
        json_end = raw.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            parsed = json.loads(raw[json_start:json_end])
            is_grounded = parsed.get("is_grounded", True)
            if not is_grounded:
                feedback = parsed.get("hallucinated_claims", ["Unspecified hallucination detected."])
    except Exception as e:
        telemetry.log_warning("Critic evaluation failed — flagging for rewrite", error=str(e))
        # Default to NOT grounded so the writer retries rather than silently
        # passing a response that was never audited. The loop_count cap (3)
        # prevents infinite rewrite cycles.
        is_grounded = False
        feedback = [f"Critic could not evaluate (error: {str(e)[:120]}). Please regenerate carefully."]
        
    duration = time.time() - start_time
    telemetry.log_info(f"Critic audit complete. Grounded: {is_grounded}. Loop: {state.get('loop_count', 0)}", duration_ms=round(duration * 1000, 2))
    
    return {"is_grounded": is_grounded, "critic_feedback": feedback}

# ── 5. Graph Builder ───────────────────────────────────────────────────
def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("router", route_node)
    workflow.add_node("analyst", retrieve_node)
    workflow.add_node("writer", generate_node)
    workflow.add_node("critic", critic_node)
    
    workflow.set_entry_point("router")
    
    def should_retrieve(state):
        if state.get("query_category", "financial") == "general":
            return "writer"
        return "analyst"
    
    workflow.add_conditional_edges("router", should_retrieve, {"analyst": "analyst", "writer": "writer"})
    workflow.add_edge("analyst", "writer")
    workflow.add_edge("writer", "critic")
    
    def check_hallucination(state):
        if state.get("query_category", "financial") == "general":
             return "pass"
        if (state.get("loop_count") or 0) >= 3:
            telemetry.log_warning("Maximum correction loops reached. Aborting.")
            return "abort"
        if state.get("is_grounded"):
            return "pass"
        return "rewrite"

    workflow.add_conditional_edges(
        "critic",
        check_hallucination,
        {
             "pass": END,
             "rewrite": "writer",
             "abort": END
        }
    )
    
    return workflow.compile()
