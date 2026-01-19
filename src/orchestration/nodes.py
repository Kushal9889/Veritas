import google.generativeai as genai
from src.orchestration.state import AgentState
from src.retrieval.query import query_financial_docs
from src.core.config import settings

# setup api key
genai.configure(api_key=settings.GOOGLE_API_KEY)

# global var for the model name 
MODEL_NAME = 'models/gemini-2.5-flash'

def get_persona(question):
    """
    elite persona selector.
    maps the user intent to a specific c-suite role.
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
    analyst node - fetches docs
    """
    print("--- 🕵️‍♂️ ANALYST: searching for info... ---")
    question = state["question"]
    
    # getting 5 docs just to be safe
    results = query_financial_docs(question, top_k=5)
    retrieved_texts = [r["text"] for r in results]
    
    return {"documents": retrieved_texts}

def generate_node(state: AgentState):
    """
    writer node - high conviction persona generation
    """
    question = state["question"]
    documents = state["documents"]
    
    # determining the elite persona
    role = get_persona(question)
    print(f"--- ✍️ WRITER: adopting persona: {role}... ---")
    
    context_block = "\n\n".join(documents)
    
    # this prompt is now tuned for "aggressive alpha" tone
    # no passive voice. no corporate fluff.
    prompt = f"""
    act as a world class {role}. 
    your goal is to give a brutal, honest, and high-value assessment.
    
    context from 10k filings:
    {context_block}
    
    user question: 
    {question}
    
    reasoning framework:
    1. analyze: what is the single biggest lever or threat here?
    2. extract: find the numbers that matter. ignore the pr speak.
    3. synthesize: build the case. why does this matter right now?
    
    output instructions:
    - start with a hook. cut straight to the chase. (e.g., "here is the reality," or "let's be clear.")
    - use strong, active verbs ("dominate," "crush," "double down," "eliminate").
    - be opinionated. if the data is good, say it's incredible. if it's bad, say it's toxic.
    - structure: use punchy bullet points.
    - final verdict: give a clear instruction on what to do next.
    
    tone: elite, high-conviction, wall street confidence.
    do not mention "context" or "documents".
    
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
        print("--- ❌ CRITIC: No documents found. ---")
        return {"grade": "not_relevant", "retry_count": state.get("retry_count", 0) + 1}
    
    docs_to_check = "\n\n".join(documents[:2]) 
    
    prompt = f"""
    you are a strict compliance auditor.
    check if the text below is relevant to the user question.
    
    user question: {question}
    
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