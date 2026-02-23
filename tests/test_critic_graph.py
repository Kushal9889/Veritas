import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.orchestration.workflow import build_graph

def test_compile():
    try:
        graph = build_graph()
        print("Graph compiled successfully!")
        print(graph.get_graph().draw_ascii())
    except Exception as e:
        print(f"Failed to compile graph: {e}")

if __name__ == "__main__":
    test_compile()
