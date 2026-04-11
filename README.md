# Multimodal Agentic RAG System

## Intelligent Knowledge Management powered by Multi-Agent AI

---

### Overview

A **multimodal Retrieval-Augmented Generation (RAG)** system built on a **multi-agent architecture**. It combines document ingestion, vector search, and agentic orchestration to provide accurate, source-cited answers from your documents and the web.

The system uses a **LangGraph supervisor** that routes queries to specialized worker agents — a **retrieval agent** for document search and a **web search agent** for real-time information — then synthesizes a final answer.

### Key Features

**Multimodal Document Processing**
- **PDFs**: Text extraction + embedded image analysis via PyMuPDF
- **Presentations**: PowerPoint (PPTX) with text, image, and full-slide extraction
- **Spreadsheets**: Excel and CSV with row-level grouping
- **Images**: Visual content embedding via Nomic vision model
- **Web Pages**: URL scraping via Selenium
- **Mixed Content**: Seamless handling of documents containing both text and images

**Multi-Agent Orchestration**
- **Supervisor Agent**: Routes queries and synthesizes final answers
- **Retrieval Agent**: Searches the vector store for relevant documents
- **Web Search Agent**: Queries the web via Tavily for real-time information
- **Configurable Instructions**: Customize supervisor behavior at runtime via API

**Three Ingestion Strategies**
- **Standard**: Direct text chunking + image embedding
- **Summarize**: VLM-powered page/slide summarization before embedding
- **Docling**: Advanced PDF chunking with picture classification and table extraction

**AI & Search**
- **Nomic Embeddings**: Vision (`nomic-embed-vision-v1.5`) and text (`nomic-embed-text-v1.5`) models
- **FAISS Vector Search**: Fast similarity search with precomputed embeddings
- **Strict Context Mode**: LLM-confirmed source filtering for higher precision
- **Configurable LLM**: Switch between Groq (cloud) and Ollama (local) at runtime

**Real-time Communication**
- **WebSocket Chat**: Streaming responses with heartbeat support
- **MCP Server**: Expose the RAG system as an MCP tool via SSE

---

### Architecture

```
src/
├── api/routes/            # FastAPI routes (chat, upload, LLM config)
├── agents/                # Supervisor, Agent wrapper, Tools
├── document_ingestion/    # Ingestion pipeline + Docling strategy
├── embeddings/            # Nomic embedding service (vision + text)
├── graph/                 # LangGraph state graph definition
├── prompts/               # System prompts for all agents
├── services/              # Chat service, Document service, LLM factory
├── state/                 # LangGraph state definition
└── vectorstore/           # FAISS vector store management

app.py                     # FastAPI application entry point
main.py                    # CLI for ingestion testing and graph visualization
mcp_server.py              # MCP server exposing the RAG system
data/                      # Raw documents for ingestion
embedded_data/             # Stored FAISS index and embeddings
```

**Agent Flow:**
```
User Query → Supervisor → Retrieval Agent / Web Search Agent → Supervisor → Final Answer
```

---

### Quick Start

**1. Prerequisites**
- Python 3.13+

**2. Setup Virtual Environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure Environment Variables**

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

| API Key | Purpose | Free Tier |
|---------|---------|-----------|
| `GROQ_API_KEY` | Powers LLM for chat and image summarization | Free with rate limits — [Groq Console](https://console.groq.com/) |
| `TAVILY_API_KEY` | Enables web search agent for real-time queries | 1,000 free requests/month — [Tavily](https://tavily.com/) |

**5. Create Data Directory**
```bash
mkdir data
```

**6. Run the Application**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**7. (Optional) LibreOffice for PPTX Summarization**

The "summarize" ingestion strategy for PPTX files requires LibreOffice to convert slides to images:
- Download from [libreoffice.org](https://www.libreoffice.org/download/download/)
- Add to system PATH (Windows: `C:\Program Files\LibreOffice\program`)

---

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `WebSocket` | `/api/v1/ws/chat/{chat_id}` | Chat with streaming responses |
| `POST` | `/api/v1/upload` | Upload documents (with strategy query param) |
| `POST` | `/api/v1/scrape` | Scrape and ingest a URL |
| `POST` | `/api/v1/supervisor/instructions` | Set custom supervisor instructions |
| `POST` | `/api/v1/llm-config` | Configure LLM provider and model |
| `GET` | `/api/v1/llm-config` | Get current LLM configuration |

---

### Technical Stack

- **Agent Framework**: LangChain, LangGraph
- **LLM Providers**: Groq (cloud), Ollama (local)
- **Embeddings**: Nomic vision + text via Transformers, PyTorch
- **Vector Search**: FAISS
- **Document Processing**: PyMuPDF, Docling, python-pptx, Pillow, Pandas
- **Web Framework**: FastAPI, WebSockets
- **Web Search**: Tavily
- **MCP**: FastMCP for tool server exposure

---

### License

This project is licensed under the GNU General Public License v3.0 — see the [LICENSE](LICENSE) file for details.
