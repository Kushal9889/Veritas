from langgraph.graph import StateGraph, END
from src.orchestration.state import AgentState
from src.orchestration.nodes import retrieve_node, generate_node

# --- THE ARCHITECT ---
# this file defines the flow of information (the "workflow")

def build_graph():
    """
    constructs the state machine graph.
    analyst -> writer -> end
    """
    # 1. initialize the graph with our state structure
    workflow = StateGraph(AgentState)
    
    # 2. add the workers (nodes)
    workflow.add_node("analyst", retrieve_node)
    workflow.add_node("writer", generate_node)
    
    # 3. define the flow (edges)
    
    # entry point: when the app starts, go straight to analyst
    workflow.set_entry_point("analyst")
    
    # after analyst works, pass the baton to the writer
    workflow.add_edge("analyst", "writer")
    
    # after writer works, the job is done
    workflow.add_edge("writer", END)
    
    # 4. compile the graph into an executable app
    app = workflow.compile()
    
    return app