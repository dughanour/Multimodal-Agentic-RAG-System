from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import AFTER load_dotenv so DATABASE_URL is available
from src.api.routes import chat
from src.api.routes import llm_config
from src.api.routes import sessions
from src.models.chat_models import create_tables

# Create database tables
create_tables()



app = FastAPI(
    title="Multimodal Agentic RAG API",
    description="API for the Multimodal Agentic RAG API",
    version="0.1.0",
)

# CORS Middleware - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    """A simple endpoint to confirm the API is running."""
    return {"message": "Welcome to the Multimodal Agentic RAG API API"}


# Include the API router for chat and document endpoints
app.include_router(chat.router, prefix="/api/v1")
app.include_router(sessions.router, prefix="/api/v1")
app.include_router(llm_config.router)


if __name__ == "__main__":
    import uvicorn
    # To run this API server, execute the following command in your terminal:
    # uvicorn src.api.main:app --reload
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


# run the api server using the following command:
    #uvicorn app:app --host 0.0.0.0 --port 8000 --reload#