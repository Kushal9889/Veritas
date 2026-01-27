import sys
import os
import re

# Ensure the root directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <--- NEW IMPORT
from pydantic import BaseModel
from src.orchestration.graph import build_graph

app = FastAPI(title="Veritas Financial Agent")

# --- 🔐 SECURITY: ENABLE CORS ---
# This allows the Frontend (Port 3000) to talk to this Backend (Port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # <--- CHANGE THIS to ["*"] for public deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("--- 🧠 BUILDING AGENT BRAIN... ---")
try:
    agent_app = build_graph()
    print("--- ✅ AGENT ACTIVE ---")
except Exception as e:
    print(f"--- ⚠️ Agent failed to load: {e} ---")
    agent_app = None

class ChatRequest(BaseModel):
    question: str
    user_id: str = "default_user"

class ChatResponse(BaseModel):
    response: str
    sources: list[str] = []

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    print(f"--- 📨 API: received: {payload.question} ---")
    
    if not agent_app:
        raise HTTPException(status_code=500, detail="Agent Brain is offline.")
    
    try:
        inputs = {"question": payload.question}
        result = agent_app.invoke(inputs)
        final_answer = result.get("generation", "No response.")
        
        # ELITE SOURCE EXTRACTION
        # Extracts "(Source: file.pdf, Page 5)" from the text
        raw_docs = result.get("documents", [])
        clean_sources = []
        
        for doc in raw_docs:
            match = re.search(r"\(Source:.*?, Page \d+\)", doc)
            if match:
                clean_sources.append(match.group(0))
            else:
                clean_sources.append("Source: 10-K Filing")

        clean_sources = list(set(clean_sources))
        
        return {
            "response": final_answer,
            "sources": clean_sources
        }
        
    except Exception as e:
        print(f"❌ API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)