from typing import Any, List, Tuple
from langchain_core.tools import Tool
from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
from src.prompts.prompts import DOCUMENT_CONFIRMATION_PROMPT



class Tools:
    """
    Holds shared instances for tools to access.
    Configure once at startup via `configure_tools(vectorstore, embedding_service)`.
    """
    vectorstore: Any = None
    embedding_service: Any = None
    llm: Any = None
    use_strict_context: bool = False


def configure_tools(vectorstore: Any, embedding_service: Any, llm: Any) -> None:
    Tools.vectorstore = vectorstore
    Tools.embedding_service = embedding_service
    Tools.llm = llm

def configure_retriever_mode(use_strict_context: bool):
    """Sets the retrieval strategy to the next tool call"""
    Tools.use_strict_context = use_strict_context

def format_document_for_agent(doc: Document, index: int) -> str:
    """
    ormat a single Document into a clean, readable string for the agent.
    
    Args:
        doc: The Document object to format
        index: The document number (1, 2, 3, etc.)
    
    Returns:
        A formatted string representation of the document

    """
    meta = doc.metadata if hasattr(doc, "metadata") else {}
    doc_type = meta.get("type", "text")
    filename = meta.get("file_name", "Unknown")
    page = meta.get("page") or meta.get("slide_number")
    is_full_page = meta.get("is_full_page", False)

    # Build formated output string
    lines = []
    lines.append(f"\━━━━━━━━━━📄 DOCUMENT {index}━━━━━━━━━")
    lines.append(f"║ Source: {filename}")

    if page is not None:
        if doc_type == "image" and is_full_page:
            lines.append(f"║ 📍 Page: {page} (Full Page Image)")
        else:
            lines.append(f"║ 📍 Page: {page}")
    
    lines.append(f"║ 📝 Type: {doc_type}")
    lines.append("")
    lines.append("Content:")
    lines.append(doc.page_content)
    lines.append("╔═════════"+ "END OF DOCUMENT" + "════════╗")  # Separator line at the end

    return " ".join(lines)



def retriever_tool_function(query: str, k: int = 6) -> str:
    """
    Retrieve top-k documents from the vectorstore for a query.
    Returns a list of Document objects with source citations embedded in page_content.
    """
    if Tools.vectorstore is None or Tools.embedding_service is None:
        return "No vectorstore or embedding service configured."

    docs_with_scores: List[Tuple[Document, float]] = Tools.vectorstore.retrieve_by_query_with_scores(
        query, Tools.embedding_service, k=k
    )
    
    if not docs_with_scores:
        return "No documents found."
    
    print(f"📚 Retrieved {len(docs_with_scores)} documents")

    # --- Strict Context Mode: Filter to single source ---
    if Tools.use_strict_context:
        print("🕵️‍♂️ Running in Strict Context mode with LLM Confirmation.")
        candidate_docs = docs_with_scores[:6]

        if candidate_docs:
            # Format candidates for LLM to choose best source
            formatted_candidates = []
            for i, (doc, score) in enumerate(candidate_docs):
                content_preview = doc.page_content[:250].strip()
                source = doc.metadata.get("source", "Unknown")
                formatted_candidates.append(f"[{i}] Source: {source}\nContent: {content_preview}...")
            
            # Ask LLM to choose the best document
            confirmation_prompt = DOCUMENT_CONFIRMATION_PROMPT.format(
                query=query,
                documents="\n\n".join(formatted_candidates)
            )
            llm_response = Tools.llm.invoke(confirmation_prompt)
            
            try:
                best_doc_index = int(llm_response.content.strip())
                confirmed_doc, _ = candidate_docs[best_doc_index]
                top_text_source = confirmed_doc.metadata.get("source")
                print(f"✅ LLM confirmed source: {top_text_source}")
            except (ValueError, IndexError):
                print("⚠️ LLM confirmation failed. Using top document.")
                top_text_source = candidate_docs[0][0].metadata.get("source")
            
            # Filter to only docs from confirmed source
            if top_text_source:
                docs_with_scores = [
                    (doc, score) for doc, score in docs_with_scores 
                    if doc.metadata.get("source") == top_text_source
                ]
    else:
        print("🌐 Running in Best Match mode.")

    # --- Process and format documents for agent ---
    formatted_docs = []
    
    for idx, (doc, score) in enumerate(docs_with_scores[:8], start=1):
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        doc_type = meta.get("type", "text")
        
        # For images: ensure we have the analysis in page_content
        if doc_type == "image":
            ingestion_strategy = meta.get("ingestion_strategy")
            
            if ingestion_strategy in ("summarize", "docling"):
                # Already has summary from ingestion - use as-is
                pass
            elif "image_base64" in meta and Tools.llm:
                # Standard strategy: analyze image on-the-fly
                file_name = meta.get("file_name", "Unknown")
                print(f"🖼️ Analyzing image from {file_name}...")
                prompt = f"Analyze this image in relation to the query: '{query}'. Describe what you see."
                multimodal_message = HumanMessage(content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{meta['image_base64']}"}}
                ])
                analysis = Tools.llm.invoke([multimodal_message])
                # Update the doc's page_content with analysis
                doc.page_content = f"[Image Analysis]: {analysis.content}"
        
        # Format using our helper function
        formatted_docs.append(format_document_for_agent(doc, idx))
    
    # Build final output string
    header = "=" * 5 + "📚 RETRIEVED DOCUMENTS" + "=" * 5
    footer = "=" * 5 + "END OF RETRIEVED DOCUMENTS" + "=" * 5
    
    result = f"{header}\n\n" + "\n".join(formatted_docs) + f"\n{footer}"
        
    return result

def retriever_tool() -> Tool:
    return Tool(
        func=retriever_tool_function,
        name="retriever_tool",
        description="Retrieve documents from the vectorstore",
    )

def web_search_tool_function(query: str) -> str:
    """Search the web for information"""
    # open source web search tool to do..
    tavily_search = TavilySearch(max_results=3)
    return tavily_search.invoke(query)


def web_search_tool() -> Tool:
    return Tool(
        func=web_search_tool_function,
        name="web_search_tool",
        description="Search the web for information",
    )