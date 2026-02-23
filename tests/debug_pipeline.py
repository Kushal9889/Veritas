import json
from src.orchestration.workflow import build_graph

def run():
    print("Building Agent Graph...")
    graph = build_graph()
    
    query = {"question": "What is the exact percentage of total revenue that Apple generated from iPhone sales in Q4 relative to their overall operating margin exposure?"}
    print(f"\nSubmitting Query: {query}")
    
    try:
        # We will iterate through the stream to capture each node's exact output
        for step in graph.stream(query):
            node_name = list(step.keys())[0]
            output = step[node_name]
            print(f"\n--- Node Executed: [ {node_name.upper()} ] ---")
            
            # Print keys safely to avoid massive console dumps of long text
            for k, v in output.items():
                if isinstance(v, list):
                    print(f"  {k} (list len={len(v)}): {v[:2]}...")
                elif isinstance(v, str):
                    print(f"  {k} (str len={len(v)}): {v[:100]}...")
                else:
                    print(f"  {k}: {v}")

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run()
