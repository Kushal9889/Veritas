import google.generativeai as genai
from src.orchestration.state import AgentState
from src.retrieval.query import query_financial_docs
from src.retrieval.graph_query import query_graph
from src.core.config import settings

genai.configure(api_key=settings.GOOGLE_API_KEY)
MODEL_NAME = 'models/gemini-2.5-flash'

def extract_entities(question):
    """
    uses the llm to figure out what to search in the graph.
    """
    prompt = f"""
    extract the key entities (companies, people, topics) from this question.
    return only a comma-separated list. no extra words.
    
    question: {question}
    """
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        entities = [e.strip() for e in response.text.split(",")]
        return entities
    except:
        return []

def retrieve_node(state: AgentState):
    """
    analyst node - hybrid search (vector + graph)
    """
    print("--- 🕵️‍♂️ ANALYST: starting hybrid search... ---")
    question = state["question"]
    
    # 1. VECTOR SEARCH (With Metadata)
    print("   📡 searching vector database...")
    # Getting the rich objects (text + page numbers)
    vector_results = query_financial_docs(question, top_k=5)
    
    # CRITICAL: We format the text to include the source tag.
    # The LLM will see: "Content... (Source: tesla.pdf, Page 5)"
    doc_texts = [f"Content: {r['text']} (Source: {r['source']}, Page {r['page']})" for r in vector_results]
    
    # 2. GRAPH SEARCH
    print("   🕸️ searching knowledge graph...")
    entities = extract_entities(question)
    graph_texts = []
    
    try:
        for entity in entities:
            connections = query_graph(entity)
            for c in connections:
                fact = f"{c['source']} is connected to {c['target']} via {c['relationship']}"
                graph_texts.append(fact)
    except:
        print("   ⚠️ graph search skipped")

    return {
        "documents": doc_texts, 
        "graph_data": graph_texts
    }

def get_persona(question):
    q = question.lower()
    if "risk" in q or "danger" in q: return "chief risk officer"
    elif "revenue" in q or "profit" in q: return "investment banker"
    elif "legal" in q: return "forensic accountant"
    else: return "senior financial analyst"

def generate_node(state: AgentState):
    """
    writer node
    """
    question = state["question"]
    documents = state["documents"]
    graph_data = state.get("graph_data", [])
    
    context_block = "--- TEXT EVIDENCE ---\n" + "\n\n".join(documents)
    
    if graph_data:
        context_block += "\n\n--- GRAPH EVIDENCE ---\n" + "\n".join(graph_data)
    
    role = get_persona(question)
    print(f"--- ✍️ WRITER: adopting persona: {role}... ---")
    
    # --- UPDATED PROMPT FOR INTELLIGENCE ---
    prompt = f"""
    You are an elite {role} with decades of experience.
    
    YOUR GOAL:
    Answer the user's question by synthesizing the provided context with your own financial expertise. 
    Do not just summarize the text. Analyze it. Offer strategic insights, critiques, or "outside the box" recommendations.
    
    RULES:
    1. **Facts must be cited:** If you quote a specific number, risk, or statement from the company, you MUST cite it using the (Source: X, Page Y) format from the evidence.
    2. **Analysis does not need citations:** When you provide your own strategic advice or reasoning based on general financial principles, do not force a citation.
    3. **Be Direct:** Do not say "The document states..." repeatedly. Just state the fact.
    
    CONTEXT:
    {context_block}
    
    USER QUESTION: 
    {question}
    """
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        return {"generation": response.text}
    except Exception as e:
        return {"generation": "Error generating response."}