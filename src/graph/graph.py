from src.agents.supervisor import Supervisor
from src.agents.agent import Agent
from src.agents.tools import retriever_tool, web_search_tool
from src.vectorstore.vectorstore import VectorStore
from src.embeddings.embedding_service import EmbeddingService
from src.prompts.prompts import RETRIEVAL_AGENT_PROMPT, WEB_SEARCH_AGENT_PROMPT
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from src.state.state import State
from langgraph.graph import StateGraph, START
from langgraph.types import Command
from typing import Literal


class Graph:
    """Graph for the agentic RAG system"""
    def __init__(self,vectorstore: VectorStore, embedding_service: EmbeddingService, llm: BaseChatModel):
        self.vectorstore = vectorstore
        self.embedding_service = embedding_service
        ## to do list.... 
        self.llm = llm
    

    def build_workers(self) -> list[Agent]:
        """Build the workers for the agentic RAG system"""

        self.retrieval_agent = Agent(agent_name="retrieval_agent", llm=self.llm, tools=[retriever_tool()],prompt=SystemMessage(content=RETRIEVAL_AGENT_PROMPT))
        self.web_search_agent = Agent(agent_name="web_search_agent", llm=self.llm, tools=[web_search_tool()],prompt=SystemMessage(content=WEB_SEARCH_AGENT_PROMPT))
        
        # Build the agents
        self.retrieval_agent.build()
        self.web_search_agent.build()

        return [self.retrieval_agent, self.web_search_agent]

    def build_supervisor(self):
        """Build the supervisor for the agentic RAG system"""

        workers = self.build_workers()
        names = [worker.agent_name for worker in workers]
        self.supervisor = Supervisor(agents=names, llm=self.llm)
        return self.supervisor.build_supervisor_node()
    
    def web_search_agent_node(self, state: State) -> Command[Literal["supervisor"]]:
        """Build the web search agent node for the agentic RAG system"""
        # Note: to show intermediate results when invoke web_search_agent
        for chunk in self.web_search_agent.stream(state):
        #This yields tool and message updates progressively
            print(f"Update from web search agent subgraph:\n{chunk}\n")
        result = self.web_search_agent.invoke(state)
        return Command(
            update={"messages":[HumanMessage(content=result["messages"][-1].content,name="search_agent")] 
            },
            # we want our workers to respond "report back" to the supervisor when done
            goto="supervisor"
            )
    
    def retrieval_agent_node(self, state: State) -> Command[Literal["supervisor"]]:
        """Build the retrieval agent node for the agentic RAG system"""
        # Note: to show intermediate results when invoke retrieval_agent
        for chunk in self.retrieval_agent.stream(state):
        # This yields tool and message updates progressively
            print(f"Update from retrieval_agent subgraph:\n{chunk}\n")
        
        result = self.retrieval_agent.invoke(state)
        
        # Get the agent's text summary
        agent_summary = result["messages"][-1].content
        

        # The final message content is the agent's summary plus the multimodal context
        final_message_content = [{"type": "text", "text": agent_summary}] 

        return Command(
            update={
                "messages": [HumanMessage(content=final_message_content, name="retrieval_agent")],
            },
            goto="supervisor"
        )
    
    def build_graph(self):
        """Build the graph for the agentic RAG system"""
        supervisor_node = self.build_supervisor()
        graph = StateGraph(State)
        graph.add_node("supervisor", supervisor_node)
        graph.add_node("retrieval_agent", self.retrieval_agent_node)
        graph.add_node("web_search_agent", self.web_search_agent_node)
        graph.add_edge(START, "supervisor")
        graph = graph.compile()
        return graph
