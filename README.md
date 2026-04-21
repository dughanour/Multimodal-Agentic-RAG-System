# Multimodal Agentic RAG System

## Intelligent Knowledge Management powered by Multi-Agent AI

---

### Overview

A **multimodal Retrieval-Augmented Generation (RAG)** system built on a **multi-agent architecture**. It combines document ingestion, vector search, and agentic orchestration to provide accurate, source-cited answers from your documents and the web.

The system uses a **LangGraph supervisor** that routes queries to specialized worker agents — a **retrieval agent** for document search and a **web search agent** for real-time information — then synthesizes a final answer.

Chat sessions and full message history are persisted in PostgreSQL, enabling conversation continuity and memory across sessions.

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
- **PostgreSQL + pgvector**: Production-grade vector search with persistent storage via Docker
- **FAISS Vector Search**: Available as a development/local alternative
- **Strict Context Mode**: LLM-confirmed source filtering for higher precision
- **Configurable LLM**: Switch between Groq (cloud) and Ollama (local) at runtime

**Chat Sessions & Conversation Memory**
- **Persistent Sessions**: Chat sessions stored in PostgreSQL with auto-generated titles
- **Message History**: Full conversation history saved and loaded per session
- **Conversation Memory**: Previous messages are injected into LLM context for multi-turn conversations
- **Session Management**: Create, list, switch, and delete sessions via REST API
- **Real-time Title Updates**: Session titles pushed to the frontend via WebSocket

**Real-time Communication**
- **WebSocket Chat**: Streaming responses with heartbeat support
- **MCP Server**: Expose the RAG system as an MCP tool via SSE

---

### Architecture

```
src/
├── api/routes/            # FastAPI routes (chat, upload, sessions, LLM config)
├── agents/                # Supervisor, Agent wrapper, Tools
├── document_ingestion/    # Ingestion pipeline + Docling strategy
├── embeddings/            # Nomic embedding service (vision + text) + LangChain adapter
├── graph/                 # LangGraph state graph definition
├── models/                # SQLAlchemy ORM models + database connection
├── prompts/               # System prompts for all agents
├── services/              # Chat service, Document service, Session service, LLM factory
├── state/                 # LangGraph state definition
└── vectorstore/           # PostgreSQL (pgvector) + FAISS vector store

app.py                     # FastAPI application entry point
main.py                    # CLI for ingestion testing and graph visualization
mcp_server.py              # MCP server exposing the RAG system
docker-compose.yml         # PostgreSQL + pgvector container definition
data/                      # Raw documents for ingestion
```

**Agent Flow:**
```
User Query → Supervisor → Retrieval Agent / Web Search Agent → Supervisor → Final Answer
```

**Data Flow:**
```
Documents → Ingestion Pipeline → Nomic Embeddings → PostgreSQL (pgvector)
User Message → WebSocket → Save to DB → Load History → LangGraph → Stream Response → Save to DB
```

---

### Quick Start

**1. Prerequisites**
- Python 3.13+
- Docker Desktop (for PostgreSQL + pgvector)

**2. Start the Database**

Launch PostgreSQL with pgvector using Docker:
```bash
docker compose up -d
```
This starts a PostgreSQL 17 container with the pgvector extension on port `5433`. Data is persisted in a Docker volume.

**3. Setup Virtual Environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

**4. Install Dependencies**
```bash
pip install -r requirements.txt
```

**5. Configure Environment Variables**

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
DATABASE_URL=postgresql+psycopg://rag_user:rag_password@localhost:5433/rag_db
```

| Variable | Purpose | Source |
|----------|---------|-------|
| `GROQ_API_KEY` | Powers LLM for chat and image summarization | Free with rate limits — [Groq Console](https://console.groq.com/) |
| `TAVILY_API_KEY` | Enables web search agent for real-time queries | 1,000 free requests/month — [Tavily](https://tavily.com/) |
| `DATABASE_URL` | PostgreSQL connection string for vector store and chat sessions | Provided by Docker Compose setup |

**6. Create Data Directory**
```bash
mkdir data
```

**7. Run the Application**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

On startup, the application automatically creates the required database tables (`chat_sessions`, `chat_messages`, and pgvector collections).

**8. (Optional) LibreOffice for PPTX Summarization**

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
| `POST` | `/api/v1/sessions` | Create a new chat session |
| `GET` | `/api/v1/sessions` | List all chat sessions |
| `GET` | `/api/v1/sessions/{session_id}/messages` | Get messages for a session |
| `DELETE` | `/api/v1/sessions/{session_id}` | Delete a session and its messages |
| `PATCH` | `/api/v1/sessions/{session_id}` | Update a session title |
| `POST` | `/api/v1/supervisor/instructions` | Set custom supervisor instructions |
| `POST` | `/api/v1/llm-config` | Configure LLM provider and model |
| `GET` | `/api/v1/llm-config` | Get current LLM configuration |

---

### Database Schema

The system uses PostgreSQL for both vector storage (via pgvector) and chat persistence:

**`chat_sessions`** — Stores chat session metadata
| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `title` | VARCHAR(255) | Auto-generated from first user message |
| `created_at` | TIMESTAMP | Session creation time |
| `updated_at` | TIMESTAMP | Last activity time |

**`chat_messages`** — Stores individual messages within sessions
| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `session_id` | UUID | Foreign key to `chat_sessions` |
| `role` | VARCHAR(20) | `user` or `assistant` |
| `content` | TEXT | Message content |
| `created_at` | TIMESTAMP | Message timestamp |

**`langchain_pg_embedding`** — Managed by langchain-postgres for vector storage

---

### Technical Stack

- **Agent Framework**: LangChain, LangGraph
- **LLM Providers**: Groq (cloud), Ollama (local)
- **Embeddings**: Nomic vision + text via Transformers, PyTorch
- **Vector Database**: PostgreSQL + pgvector (production), FAISS (development)
- **Chat Persistence**: PostgreSQL via SQLAlchemy ORM
- **Document Processing**: PyMuPDF, Docling, python-pptx, Pillow, Pandas
- **Web Framework**: FastAPI, WebSockets
- **Web Search**: Tavily
- **Containerization**: Docker, Docker Compose
- **MCP**: FastMCP for tool server exposure

---

### Docker Services

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `postgres` | `pgvector/pgvector:pg17` | `5433:5432` | Vector storage + chat session persistence |

Manage the database:
```bash
docker compose up -d      # Start
docker compose down        # Stop (data preserved in volume)
docker compose down -v     # Stop and delete all data
```

---

### License

This project is licensed under the GNU General Public License v3.0 — see the [LICENSE](LICENSE) file for details.
