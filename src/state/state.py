from langgraph.graph import StateGraph,MessagesState
from typing import Annotated, Any, List
import operator
from langchain_core.documents import Document

class State(MessagesState):
    """State for the agentic RAG system"""
    next: str
    # A turn counter
    turns: Annotated[int, operator.add]
    # Keep track of retrieved documents
    retrieved_docs: Annotated[List[Document], operator.add]
