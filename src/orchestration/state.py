import operator
from typing import Annotated, List, TypedDict

# defining the state of our agent
# this dict holds data moving between nodes
class AgentState(TypedDict):
    question: str
    documents: List[str]     # Vector DB results (text chunks)
    graph_data: List[str]    # NEW: Graph DB results (relationships)
    generation: str
    grade: str
    retry_count: int