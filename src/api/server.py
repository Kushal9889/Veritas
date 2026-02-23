import sys
import os
import re

# Ensure the root directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.orchestration.workflow import build_graph
from src.retrieval.search import warmup
from src.core.telemetry import get_telemetry
import json

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

telemetry = get_telemetry("api.server")

print("--- 🧠 BUILDING AGENT BRAIN... ---")
try:
    telemetry.log_info("Initializing agent graph")
    agent_app = build_graph()
    
    telemetry.log_info("Warming up cross-encoder models")
    warmup()
    
    print("--- ✅ AGENT ACTIVE ---")
except Exception as e:
    telemetry.log_error("Agent failed to load", error=e)
    print(f"--- ⚠️ Agent failed to load: {e} ---")
    agent_app = None

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat_endpoint(payload: ChatRequest):
    telemetry.log_info("Incoming chat request", question=payload.question)
    
    if not agent_app:
        raise HTTPException(status_code=500, detail="Agent Brain is offline.")
    
    async def generate_stream():
        try:
            import asyncio
            from langchain_core.runnables import RunnableConfig
            
            # Use LangGraph for LangSmith tracing
            full_response = ""
            sources = []
            
            # Streaming LangGraph trace end-to-end
            langsmith_config = RunnableConfig(
                tags=["fastapi_chat", "production"], 
                run_name="Veritas_Streaming_Chat"
            )
            
            for chunk in agent_app.stream({"question": payload.question}, config=langsmith_config):
                if "analyst" in chunk:
                    # Extract sources from analyst node
                    docs = chunk["analyst"].get("documents", [])
                    for doc in docs:
                        match = re.search(r"^\[[^\]]+\]", doc)
                        if not match:
                            match = re.search(r"\(Source:.*?, Page \d+\)", doc)
                        if match:
                            sources.append(match.group(0))
                    
                    # Send sources immediately
                    yield f"data: {json.dumps({'type': 'sources', 'sources': list(set(sources))})}\n\n"
                    await asyncio.sleep(0.01)
                
                if "writer" in chunk:
                    # Get the generated response
                    generation = chunk["writer"].get("generation", "")
                    
                    # Stream the new text
                    if generation and generation != full_response:
                        new_text = generation[len(full_response):]
                        full_response = generation
                        
                        # Split into smaller chunks for smoother streaming
                        for char in new_text:
                            yield f"data: {json.dumps({'type': 'content', 'text': char})}\n\n"
                            await asyncio.sleep(0.001)  # Very small delay for smooth streaming
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            print(f"❌ API Error: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(generate_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
