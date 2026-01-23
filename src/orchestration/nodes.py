import google.generativeai as genai
from src.orchestration.state import AgentState
from src.retrieval.query import query_financial_docs
from src.retrieval.graph_query import query_graph
from src.core.config import settings

# setup api key
genai.configure(api_key=settings.GOOGLE_API_KEY)

MODEL_NAME = 'models/gemini-2.5-flash'

def extract_entities(question):
    """
    uses the llm to figure out what to search in the graph.
    input: "what are the risks for tesla?"
    output: ["tesla", "risk"]
    """
    prompt = f"""
    extract the key entities (companies, people, topics) from this question.
    return only a comma-separated list. no extra words.
    
    question: {question}
    """
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        # cleaning up the response
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
    
    # 1. VECTOR SEARCH (The "Fuzzy" Memory)
    print("   📡 searching vector database...")
    vector_results = query_financial_docs(question, top_k=5)
    vector_texts = [r["text"] for r in vector_results]
    
    # 2. GRAPH SEARCH (The "Structured" Memory)
    print("   🕸️ searching knowledge graph...")
    # ask ai what to look for
    entities = extract_entities(question)
    print(f"      > identified entities: {entities}")
    
    graph_texts = []
    for entity in entities:
        # querying neo4j for each entity
        connections = query_graph(entity)
        for c in connections:
            # formatting it as a sentence for the writer to read
            # e.g. "Tesla --[SUPPLIED_BY]--> Panasonic"
            fact = f"{c['source']} is connected to {c['target']} via {c['relationship']}"
            graph_texts.append(fact)
            
    return {
        "documents": vector_texts,
        "graph_data": graph_texts
    }

def get_persona(question):
    """
    helper to pick the best role based on the user question.
    """
    q = question.lower()
    
    if "risk" in q or "danger" in q:
        return "chief risk officer"
    elif "revenue" in q or "profit" in q or "invest" in q:
        return "investment banker"
    elif "legal" in q or "audit" in q:
        return "forensic accountant"
    else:
        return "senior financial analyst"

def generate_node(state: AgentState):
    """
    writer node - synthesizes vector + graph data
    """
    question = state["question"]
    documents = state["documents"]
    graph_data = state.get("graph_data", [])
    
    # combining both memory sources
    context_block = "--- TEXT EVIDENCE (VECTOR DB) ---\n" + "\n\n".join(documents)
    
    if graph_data:
        context_block += "\n\n--- STRUCTURAL FACTS (KNOWLEDGE GRAPH) ---\n" + "\n".join(graph_data)
    
    role = get_persona(question)
    print(f"--- ✍️ WRITER: adopting persona: {role}... ---")
    
    prompt = f"""
    you are a elite {role}. 
    use the hybrid context below to answer the user question.
    
    combine the text details with the structural facts from the graph.
    if the graph shows a connection (like a supplier or risk), highlight it.
    
    CONTEXT:
    {context_block}
    
    USER QUESTION: 
    {question}
    
    ANSWER:
    """
    
    generation_config = genai.types.GenerationConfig(temperature=0.3)
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt, generation_config=generation_config)
        return {"generation": response.text}
    except Exception as e:
        print(f"❌ Generation Error: {e}")
        return {"generation": "error generating response"}

def grade_documents_node(state: AgentState):
    """
    critic node - same as before
    """
    print("--- ⚖️ CRITIC: Grading documents... ---")
    question = state["question"]
    documents = state["documents"]
    
    if not documents:
        return {"grade": "not_relevant", "retry_count": state.get("retry_count", 0) + 1}
    
    docs_to_check = "\n\n".join(documents[:2]) 
    
    prompt = f"""
    you are a strict compliance auditor.
    check if relevant.
    question: {question}
    text: {docs_to_check}
    reply 'relevant' or 'not_relevant' only.
    """
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        grade = response.text.strip().lower()
    except:
        grade = "relevant" 
        
    if "not_relevant" in grade:
        print("--- ❌ CRITIC: Documents irrelevant. Retrying... ---")
        return {"grade": "not_relevant", "retry_count": state.get("retry_count", 0) + 1}
    else:
        print("--- ✅ CRITIC: Documents are relevant. ---")
        return {"grade": "relevant"}