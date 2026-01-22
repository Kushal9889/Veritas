import google.generativeai as genai
from src.orchestration.state import AgentState
from src.retrieval.query import query_financial_docs
from src.core.config import settings
from src.tools.search import search_web 
# ^ removed get_stock_price from import

# setup api key
genai.configure(api_key=settings.GOOGLE_API_KEY)

# global var for the model name 
MODEL_NAME = 'models/gemini-2.5-flash'

def get_persona(question):
    """
    elite persona selector.
    """
    q = question.lower()
    
    if any(x in q for x in ["risk", "danger", "threat", "lawsuit", "sue", "compliance"]):
        return "chief risk officer (cro)"
    elif any(x in q for x in ["revenue", "profit", "growth", "invest", "stock", "margin", "earnings"]):
        return "aggressive hedge fund manager"
    elif any(x in q for x in ["audit", "accounting", "fraud", "report", "filing", "10-k", "sheet"]):
        return "forensic accountant"
    elif any(x in q for x in ["compet", "market", "strategy", "moat", "position", "share"]):
        return "activist investor"
    elif any(x in q for x in ["supply", "factory", "product", "manufactur", "logistics", "scale"]):
        return "vp of operations"
    elif any(x in q for x in ["debt", "cash", "liquidity", "bankrupt", "loan", "bond", "interest"]):
        return "distressed debt specialist"
    elif any(x in q for x in ["tech", "patent", "r&d", "innovat", "software", "ai"]):
        return "chief technology officer"
    else:
        return "senior wall street analyst"

def retrieve_node(state: AgentState):
    """
    analyst node - fetches docs AND web data
    """
    print("--- 🕵️‍♂️ ANALYST: gathering intel... ---")
    question = state["question"]
    
    # 1. search internal pdfs (the past)
    results = query_financial_docs(question, top_k=3)
    internal_text = "\n".join([r["text"] for r in results])
    
    # 2. search live web (the present)
    # simple logic: if it looks like a timely question, hit the web
    web_data = search_web(question + " financial news current")
    
    # removed the stock price logic here. purely text based now.
    
    # combine it all
    combined_docs = [
        f"--- INTERNAL 10-K DATA ---\n{internal_text}",
        f"--- LIVE WEB SEARCH ---\n{web_data}"
    ]
    
    return {"documents": combined_docs}

def generate_node(state: AgentState):
    """
    writer node - high conviction persona generation
    """
    question = state["question"]
    documents = state["documents"]
    
    role = get_persona(question)
    print(f"--- ✍️ WRITER: adopting persona: {role}... ---")
    
    context_block = "\n\n".join(documents)
    
    prompt = f"""
    act as a world class {role}. 
    you have access to internal filings (past) and live web data (present).
    your goal is to synthesize these into a killer insight.
    
    data streams:
    {context_block}
    
    user question: 
    {question}
    
    reasoning framework:
    1. compare: does the live news contradict the old 10-k?
    2. analyze: ignore the fluff. focus on the discrepancies.
    3. synthesize: give a forward-looking recommendation.
    
    output instructions:
    - start with a hook. cut straight to the chase.
    - use strong, active verbs.
    - if you see live news, mention it explicitly ("breaking news suggests...").
    - final verdict: buy, sell, or hold?
    
    tone: elite, high-conviction, wall street confidence.
    
    answer:
    """
    
    generation_config = genai.types.GenerationConfig(
        temperature=0.3
    )
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt, generation_config=generation_config)
        return {"generation": response.text}
    except Exception as e:
        print(f"❌ Generation Error: {e}")
        return {"generation": "error generating response"}

def grade_documents_node(state: AgentState):
    """
    critic node - checks if docs are relevant
    """
    print("--- ⚖️ CRITIC: Grading documents... ---")
    
    question = state["question"]
    documents = state["documents"]
    
    if not documents:
        return {"grade": "not_relevant", "retry_count": state.get("retry_count", 0) + 1}
    
    # quick check
    docs_to_check = "\n\n".join(documents)[:2000]
    
    prompt = f"""
    you are a strict compliance auditor.
    check if the text below is relevant to the user question: "{question}"
    
    retrieved text:
    {docs_to_check}
    
    reply only with the word "relevant" or "not_relevant".
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