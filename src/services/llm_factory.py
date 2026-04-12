from typing import Optional
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

class LLMFactory:
    """Factory class for creating LLM instances."""
    def __init__(
        self,
        default_provider: str = "groq",
        default_groq_model: str = "meta-llama/llama-4-maverick-17b-128e-instruct",
        default_groq_model2: str = "openai/gpt-oss-20b",
        default_groq_model3: str = "openai/gpt-oss-120b",
        default_ollama_model: str = "ministral-3:3b",
        ollama_base_url: str = "http://localhost:11434",
        ):

        self.default_provider = default_provider
        self.default_groq_model = default_groq_model
        self.default_groq_model2 = default_groq_model2
        self.default_groq_model3 = default_groq_model3
        self.default_ollama_model = default_ollama_model
        self.ollama_base_url = ollama_base_url
    
        # In-memory config storage
        self._config = {
            "provider": default_provider,
            "model": None,
            "api_key": None,
        }
    
    def set_config(self, provider: str, model: Optional[str] = None, api_key: Optional[str] = None):
        """Set the configuration for the LLM."""
        self._config["provider"] = provider
        self._config["model"] = model
        self._config["api_key"] = api_key
    
    def get_config(self):
        """Get the current configuration."""
        return self._config.copy()
    
    def get_llm(self, provider: Optional[str] = None, model: Optional[str] = None, api_key: Optional[str] = None):
        """Get an LLM instance based on the provider and model."""
        provider = (provider or self._config["provider"] or self.default_provider).lower()
        model = model or self._config["model"]
        api_key = api_key or self._config["api_key"]

        if provider == "ollama":
            return ChatOllama(
            model=model or self.default_ollama_model,
            base_url=self.ollama_base_url,
            temperature=0.0,
            )
        
        
        # default: groq
        return ChatGroq(
            model_name=model or self.default_groq_model3,
            api_key=api_key,
            temperature=0.0,
        )
        
# Singleton instance for shared access
llm_factory = LLMFactory()

