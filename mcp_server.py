from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

from src.services.chat_service import ChatService
from langchain_core.messages import HumanMessage

# Initialize MCP server with SSE transport settings
# host="0.0.0.0" allows connections from other machines
mcp = FastMCP("Multimodal Agentic RAG", host="0.0.0.0", port=8000)

# Initialize the ChatService (singleton - loads graph once)
chat_service = ChatService()

@mcp.tool()
def ask_multimodal_rag(query: str) -> str:
    """
    Query the Multimodal Agentic RAG (Retrieval-Augmented Generation) knowledge base.
    
    This tool provides intelligent question-answering over a curated document repository
    containing PDFs, PowerPoint presentations, Excel spreadsheets, CSV files, images, 
    and web-scraped content. It uses a multi-agent system with specialized retrieval 
    and web search capabilities to find and synthesize accurate answers.
    
    Use this tool when you need to:
    - Answer questions about company documents, reports, or presentations
    - Extract specific data, statistics, or facts from uploaded files
    - Analyze charts, tables, diagrams, or images within documents
    - Find information across multiple document types simultaneously
    - Get insights from structured data (Excel/CSV) with proper context
    
    The system automatically:
    - Retrieves the most relevant document chunks using semantic search
    - Analyzes images and visual content when relevant to the query
    - Cites sources with filename and page/slide numbers
    - Falls back to web search if local documents don't contain the answer
    
    Args:
        query: A natural language question or request. Be specific and detailed 
        
    Returns:
        A comprehensive answer synthesized from retrieved documents, including
        source citations (filename and page/slide number) for verification.
        Returns an error message if no relevant information is found.
    """
    runnable = chat_service.get_runnable()
    
    input_data = {"messages": [HumanMessage(content=query)]}
    config = {"recursion_limit": 150}
    
    # Invoke synchronously and extract final answer
    result = runnable.invoke(input_data, config=config)
    
    # Get the last message from supervisor
    final_message = result["messages"][-1].content
    return final_message.replace("Final Answer:", "").strip()

if __name__ == "__main__":
    # Run with SSE transport for n8n compatibility
    mcp.run(transport="sse")

