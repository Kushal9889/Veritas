import sys
import os

# fixing path issues so we can import src
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.orchestration.state import AgentState
from src.orchestration.nodes import retrieve_node, generate_node

def test_workflow():
    # making a dummy state
    initial_state = AgentState(
        question="What are the primary risk factors?",
        documents=[],
        generation=""
    )
    
    # step 1: run the analyst
    print("\n testing analyst node...")
    update1 = retrieve_node(initial_state)
    print(f"✅ found {len(update1['documents'])} docs")
    
    # manually updating state
    initial_state["documents"] = update1["documents"]
    
    # step 2: run the writer
    print("\n testing writer node...")
    update2 = generate_node(initial_state)
    
    print("✅ generated answer:")
    print(f"'{update2['generation'][:200]}...'") 

if __name__ == "__main__":
    test_workflow()