import sys
import os

# Ensure the root directory is in the path so we can import 'src' modules
# This fixes the "ModuleNotFoundError" when running from inside subfolders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.orchestration.graph import build_graph

# initializing the app
app = FastAPI(
    title="Veritas Financial Agent",
    description="The Elite RAG Pipeline (FastAPI + LangGraph + Neo4j)",
    version="1.0.0"
)

# --- THE BRAIN ---
# Build the graph ONCE when the server starts.
# This compiles the state machine so it's ready for high-speed requests.
print("--- 🧠 BUILDING AGENT BRAIN... ---")
try:
    agent_app = build_graph()
    print("--- ✅ AGENT ACTIVE AND READY ---")
except Exception as e:
    print(f"--- ❌ CRITICAL: Failed to build graph: {e} ---")
    agent_app = None

# defining the input schema
class ChatRequest(BaseModel):
    question: str
    user_id: str = "default_user"

# defining the output schema
class ChatResponse(BaseModel):
    response: str
    sources: list[str] = []

@app.get("/health")
def health_check():
    return {"status": "active", "system": "veritas-elite"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    print(f"--- 📨 API: received question: {payload.question} ---")
    
    if not agent_app:
        raise HTTPException(status_code=500, detail="Agent Brain is offline. Check server logs.")
    
    try:
        # 1. Prepare inputs for the agent
        inputs = {"question": payload.question}
        
        # 2. Invoke the agent (Run the graph)
        # using invoke() to pass the question through the Analyst -> Writer flow
        result = agent_app.invoke(inputs)
        
        # 3. Extract the answer
        final_answer = result.get("generation", "No response generated.")
        
        # 4. Extract sources (optional - from vector docs)
        # giving the frontend a peek at what docs we used
        raw_docs = result.get("documents", [])
        sources = [doc[:200] + "..." for doc in raw_docs]
        
        return {
            "response": final_answer,
            "sources": sources
        }
        
    except Exception as e:
        print(f"❌ API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # running on 0.0.0.0 allows external access (e.g. from a frontend app)
    uvicorn.run(app, host="0.0.0.0", port=8000)