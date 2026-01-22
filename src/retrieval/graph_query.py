import sys
from langchain_community.graphs import Neo4jGraph
from src.core.config import settings

# connecting to the db
graph = Neo4jGraph(
    url=settings.NEO4J_URI,
    username=settings.NEO4J_USERNAME,
    password=settings.NEO4J_PASSWORD
)

def query_graph(entity_name):
    """
    dynamic lookup. pass any company/person name here.
    it finds the node and sees what it connects to.
    """
    print(f"--- 🕸️ GRAPH QUERY: looking up '{entity_name}'... ---")
    
    # cypher query to find connections
    # using toLower so capitalization doesnt matter (apple vs Apple)
    query = f"""
    MATCH (n)-[r]->(m)
    WHERE toLower(n.id) CONTAINS toLower('{entity_name}')
    RETURN n.id AS source, type(r) AS relationship, m.id AS target
    LIMIT 20
    """
    
    try:
        results = graph.query(query)
        
        if not results:
            print(f"--- ❌ no connections found for '{entity_name}' ---")
            return []
            
        print(f"--- ✅ found {len(results)} connections for {entity_name}! ---")
        for r in results:
            print(f"   🔹 {r['source']} --[{r['relationship']}]--> {r['target']}")
            
        return results
        
    except Exception as e:
        print(f"error querying graph: {e}")
        return []

if __name__ == "__main__": 
    # check if user passed an arg in command line
    if len(sys.argv) > 1:
        # join args in case name has spaces like "tim cook"
        target = " ".join(sys.argv[1:])
        query_graph(target)
    else:
        # fallback - ask user dynamically
        print("enter the entity you want to search (company, person, risk):")
        target = input("> ").strip()
        if target:
            query_graph(target)
        else:
            print("you didnt type anything. exiting.")