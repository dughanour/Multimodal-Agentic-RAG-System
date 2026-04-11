from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.services.llm_factory import llm_factory
from src.services.chat_service import chat_service

router = APIRouter(prefix="/api/v1", tags=["LLM Config"])

class LLMConfigRequest(BaseModel):
    provider: str # "groq" or "ollama"
    model: Optional[str] = None
    api_key: Optional[str] = None

class LLMConfigResponse(BaseModel):
    status: str
    provider: str
    model: Optional[str]

@router.post("/llm-config", response_model=LLMConfigResponse)
async def set_llm_config(config: LLMConfigRequest):
    """
    Configure the LLM provider and model.
    - provider: "groq" (cloud) or "ollama" (local)
    - model: model name (optional, uses default if not provided)
    - api_key: required for cloud providers like Groq
    """
    provider = config.provider.lower()

    #validate provider
    if provider not in ["groq", "ollama"]:
        raise HTTPException(status_code=400, detail="Invalid provider, Use 'groq' or 'ollama'.")
        
    #validate api_key for cloud providers
    if provider == "groq" and not config.api_key:
        raise HTTPException(status_code=400, detail="API key is required for Groq")
    
    # set the config
    llm_factory.set_config(
        provider=provider,
        model=config.model,
        api_key=config.api_key
        )

    # Rebuild the graph with new LLM
    chat_service.rebuild_with_new_llm()
    
    return LLMConfigResponse(
        status="LLM configuration updated successfully",
        provider=provider,
        model=config.model
    )

@router.get("/llm-config", response_model=LLMConfigResponse)
async def get_llm_conifg():
    """Get current LLM configuration."""
    config = llm_factory.get_config()
    return LLMConfigResponse(
        status="Current LLM configuration",
        provider=config["provider"],
        model=config["model"]
    )