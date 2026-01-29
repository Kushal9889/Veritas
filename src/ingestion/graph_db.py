import os
import json
from langchain_community.graphs import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from src.core.config import settings

# connecting to neo4j
graph = None
try:
    if settings.NEO4J_URI:
        print(f"--- 🕸️ Connecting to Neo4j... ---")
        graph = Neo4jGraph(
            url=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD
        )
        graph.query("RETURN 1") # Test connection
        print("--- ✅ Graph DB Online ---")
except Exception as e:
    print(f"--- ⚠️ Graph DB Offline. Running in Lite Mode. ---")
    graph = None

def extract_graph_from_text(llm, text):
    """
    manual extraction with few-shot prompting.
    giving it multiple examples so it knows how to handle complex financial data.
    """
    prompt = f"""
    you are an expert data engineer. extract knowledge graph entities and relationships from the text below.
    
    return STRICT JSON only. no markdown. no code blocks. just the raw json string.
    
    EXAMPLES:
    
    Input: "Tim Cook is the CEO of Apple, which designs products in California."
    Output: {{
      "nodes": [
        {{"id": "Tim Cook", "type": "Person"}},
        {{"id": "Apple", "type": "Organization"}},
        {{"id": "California", "type": "Location"}}
      ],
      "edges": [
        {{"source": "Tim Cook", "target": "Apple", "type": "CEO_OF"}},
        {{"source": "Apple", "target": "California", "type": "OPERATES_IN"}}
      ]
    }}
    
    Input: "Tesla relies on Panasonic for battery cells, creating a supply chain risk."
    Output: {{
      "nodes": [
        {{"id": "Tesla", "type": "Organization"}},
        {{"id": "Panasonic", "type": "Organization"}},
        {{"id": "Supply Chain Risk", "type": "Risk"}}
      ],
      "edges": [
        {{"source": "Tesla", "target": "Panasonic", "type": "SUPPLIED_BY"}},
        {{"source": "Tesla", "target": "Supply Chain Risk", "type": "EXPOSED_TO"}}
      ]
    }}
    
    Input: "The company faces regulatory challenges in the European Union regarding data privacy."
    Output: {{
      "nodes": [
        {{"id": "Company", "type": "Organization"}},
        {{"id": "European Union", "type": "Location"}},
        {{"id": "Regulatory Challenges", "type": "Risk"}}
      ],
      "edges": [
        {{"source": "Company", "target": "European Union", "type": "OPERATES_IN"}},
        {{"source": "Company", "target": "Regulatory Challenges", "type": "FACES_RISK"}}
      ]
    }}
    
    ACTUAL TEXT TO ANALYZE:
    {text}
    """
    
    try:
        # asking gemini
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        # cleaning up if the model added ```json ``` blocks
        if "```" in content:
            content = content.replace("```json", "").replace("```", "")
            
        data = json.loads(content)
        return data
    except Exception as e:
        print(f"⚠️ parsing error on chunk: {e}")
        return {"nodes": [], "edges": []}

def ingest_into_graph(documents):
    """
    Main function to process docs and save to Neo4j.
    """
    # SAFETY CHECK: If graph is None, stop here.
    if not graph:
        print("--- ⚠️ Skipping Graph Ingestion (DB is Offline) ---")
        return

    print(f"--- 🕸️ GRAPH: Starting ingestion for {len(documents)} chunks... ---")
    
    # FIX: Use 'gemini-1.5-flash' (2.5 is a typo/unavailable)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0
    )
    
    print("--- ⏳ Transforming text to graph... ---")
    
    graph_documents = []
    
    for i, doc in enumerate(documents):
        # logging progress
        print(f"   > Processing chunk {i+1}/{len(documents)}...")
        
        data = extract_graph_from_text(llm, doc.page_content)
        
        if not data.get("nodes"):
            continue

        nodes = [Node(id=n["id"], type=n["type"]) for n in data.get("nodes", [])]
        edges = [
            Relationship(
                source=Node(id=e["source"], type="Unknown"), 
                target=Node(id=e["target"], type="Unknown"),
                type=e["type"]
            )
            for e in data.get("edges", [])
        ]
        
        if nodes or edges:
            gd = GraphDocument(nodes=nodes, relationships=edges, source=doc)
            graph_documents.append(gd)
            
    print(f"--- ✅ Extracted {len(graph_documents)} graph structures. Saving to DB... ---")
    
    if graph_documents:
        try:
            graph.add_graph_documents(graph_documents)
            print("--- 🚀 Graph ingestion complete. Check Neo4j Browser. ---")
        except Exception as e:
            print(f"--- ❌ Failed to write to Neo4j: {e} ---")
    else:
        print("--- ⚠️ No graph data found. ---")