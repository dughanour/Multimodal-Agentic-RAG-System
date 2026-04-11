from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from src.graph.graph import Graph
from src.agents.tools import configure_tools
from src.vectorstore.vectorstore import VectorStore
from src.embeddings.embedding_service import EmbeddingService
from langchain_core.runnables import ConfigurableField
from src.services.llm_factory import llm_factory
import os

class ChatService:
    """
    A singleton service for managing and interacting with the LangGraph agent.
    This ensures that the graph and its components are initialized only once.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        Singleton pattern to ensure only one instance of the ChatService is created.
        “Call the __new__ method of the parent of ChatService (which is object),
        and tell it to create an instance of cls (i.e., ChatService).”
        """
        if not cls._instance:
            cls._instance = super(ChatService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Initialize only once
        if not hasattr(self, 'initialized'):
            print("🚀 Initializing ChatService and building the agentic graph...")
            self.embedding_service = EmbeddingService()
            self.vectorstore = self._load_vectorstore()
            self.llm = self._initialize_llms()
            
            # Configure the tools with the necessary components
            configure_tools(
                vectorstore=self.vectorstore, 
                embedding_service=self.embedding_service, 
                llm=self.llm
            )
            
            # Build the LangGraph runnable agent
            self._build_graph()
            self.initialized = True
            print("✅ ChatService initialized successfully.")

    def _load_vectorstore(self) -> VectorStore:
        """Loads the FAISS vector store from the local directory."""
        vs = VectorStore()
        if os.path.exists("./embedded_data/faiss_index"):
            vs.load_local("./embedded_data")
        else:
            # In a real API, you might want to handle this case differently,
            # e.g., by initializing an empty store or returning an error.
            print("⚠️ Warning: No existing FAISS index found.")
        return vs

    def _initialize_llms(self):
        """Initializes the language model using LLMFactory."""
        base_llm = llm_factory.get_llm()
        
        # Make the temperature configurable
        configurable_llm = base_llm.configurable_fields(
            temperature=ConfigurableField(
                id="temperature",
                name="LLM Temperature",
                description="The temperature of the LLM.",
            )
        )
        
        return configurable_llm
        # Note: The Ollama model is used within the tool for image summarization,
        # so it's initialized there.

    def _build_graph(self):
        """Build the LangGraph runnable agent."""
        graph_builder = Graph(
            vectorstore=self.vectorstore,
            embedding_service=self.embedding_service,
            llm=self.llm,
        )
        self.runnable = graph_builder.build_graph()
    
    def rebuild_with_new_llm(self):
        """Rebuild the graph with updated LLM config (call after llm_factory.set_config)."""
        print("🔄 Rebuilding ChatService with new LLM config...")
        self.llm = self._initialize_llms()

        # Re-configure tools with new LLM
        configure_tools(
            vectorstore=self.vectorstore,
            embedding_service=self.embedding_service,
            llm=self.llm,
        )

        # Rebuild the graph
        self._build_graph()
        print("✅ ChatService rebuilt successfully.")

    def get_runnable(self):
        """Returns the compiled LangGraph agent."""
        return self.runnable
    
    def get_llm(self):
        """Returns the initialized LLM."""
        return self.llm

# Create a single instance of the service to be used by the API
chat_service = ChatService()

def get_chat_service():
    """Dependency injector for FastAPI."""
    return chat_service
