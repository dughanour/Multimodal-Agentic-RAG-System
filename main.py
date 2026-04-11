from src.document_ingestion.document_ingestion_pipeline import DocumentIngestionPipeline
from src.embeddings.embedding_service import EmbeddingService
from src.vectorstore.vectorstore import VectorStore
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from src.graph.graph import Graph
from src.agents.tools import configure_tools, configure_retriever_mode
from dotenv import load_dotenv
import os


load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")


def view_all_documents_in_vectorstore():
    """
    Loads the vector store from disk and prints all documents it contains.
    """
    print("--- LOADING VECTOR STORE TO VIEW DOCUMENTS ---")
    vs = VectorStore()
    persist_directory = "./embedded_data"

    if os.path.exists(os.path.join(persist_directory, "faiss_index")):
        vs.load_local(persist_directory)
        
        if not vs.all_docs:
            print("Vector store loaded, but it contains no documents.")
            return

        print(f"Found {len(vs.all_docs)} documents in the vector store.\n")
        
        for i, doc in enumerate(vs.all_docs):
            print(f"--- DOCUMENT {i+1} ---")
            
            # Make a copy of metadata to modify it for printing
            metadata_to_print = doc.metadata.copy()
            if metadata_to_print.get("type") == "image" and "image_base64" in metadata_to_print:
                metadata_to_print["image_base64"] = "<base64_string>"
            
            print(f"  - Metadata: {metadata_to_print}")
            print(f"  - Content: {doc.page_content}")
            print("-" * 20 + "\n")
    else:
        print("No vector store found. Please run the ingestion process first.")
    print("--- FINISHED VIEWING DOCUMENTS ---")


def view_first_10_documents():
    """
    Loads the vector store from disk and prints the first 10 documents
    with full content and metadata for parsing inspection.
    """
    print("--- LOADING VECTOR STORE TO INSPECT FIRST 10 DOCUMENTS ---")
    vs = VectorStore()
    persist_directory = "./embedded_data"

    if not os.path.exists(os.path.join(persist_directory, "faiss_index")):
        print("No vector store found. Please run the ingestion process first.")
        return

    vs.load_local(persist_directory)

    if not vs.all_docs:
        print("Vector store loaded, but it contains no documents.")
        return

    total = len(vs.all_docs)
    limit = min(10, total)
    print(f"Showing first {limit} of {total} documents.\n")

    for i, doc in enumerate(vs.all_docs[:limit]):
        print(f"{'='*60}")
        print(f"  DOCUMENT {i+1}/{limit}")
        print(f"{'='*60}")

        metadata_to_print = doc.metadata.copy()
        if metadata_to_print.get("type") == "image" and "image_base64" in metadata_to_print:
            metadata_to_print["image_base64"] = f"<base64, {len(doc.metadata['image_base64'])} chars>"

        print(f"  Source  : {metadata_to_print.get('source', 'N/A')}")
        print(f"  Type    : {metadata_to_print.get('type', 'N/A')}")
        print(f"  Page    : {metadata_to_print.get('page', 'N/A')}")
        print(f"  All Meta: {metadata_to_print}")
        print(f"  Content ({len(doc.page_content)} chars):")
        print(f"  {'-'*56}")
        print(f"  {doc.page_content}")
        print()

    print("--- FINISHED INSPECTING FIRST 10 DOCUMENTS ---")


if __name__ == "__main__":
    # To view all documents, uncomment the line below and run: python main.py
    # view_all_documents_in_vectorstore()

    # --- Interactive Mode ---
    mode = input("Enter execution mode ('doc', 'doc10', or 'pipe'): ").strip().lower()

    if mode == 'doc':
        view_all_documents_in_vectorstore()

    elif mode == 'doc10':
        view_first_10_documents()

    elif mode == 'pipe':
        print("--- RUNNING FULL INGESTION & TEST PIPELINE ---")
        # --- Original main script logic ---
        # 1) Ingestion + embeddings
        embedding_service = EmbeddingService()
        pipeline = DocumentIngestionPipeline()
        vs = VectorStore()

        # Load existing index if present
        if os.path.exists("./embedded_data/faiss_index"):
            vs.load_local("./embedded_data")


        # --- Control Variable for Ingestion Strategy ---
        USE_SUMMARIZATION_STRATEGY = True

        if USE_SUMMARIZATION_STRATEGY:
            print("🚀 e-emptive image summarization strategy for ingestion.")
            # We need a multimodal LLM for summarization during ingestion
            llm_for_summarization = ChatGroq(temperature=0, groq_api_key=groq_api_key,model_name="meta-llama/llama-4-maverick-17b-128e-instruct")
            combined_docs, embeddings = pipeline.process_and_embed_with_summaries(
                sources=["data/nour.pptx","data/multimodal_sample.pdf","data/self_attention.pdf"],
                embedding_service=embedding_service,
                llm=llm_for_summarization
            )
        else:
            print("🚀 Using standard multimodal embedding strategy for ingestion.")
            combined_docs, embeddings = pipeline.process_and_embed(
                sources=["data/nour.pptx"],
                embedding_service=embedding_service,
            )


        print(f"Here is the combined_docs ////// {combined_docs}")
        print("\n\n")
        print("\n\n")
        print(f"Here is the embeddings ////// {embeddings}")
        print("\n\n")
        print("\n\n")

        # 2) Build vector store
        if len(combined_docs) > 0:
            vs.load_precomputed(combined_docs, embeddings)
            vs.create_vectorstore()
            vs.save_local("./embedded_data")

        else:
            print("No new documents added — using existing FAISS index.")


        # 3) Setup the agentic RAG graph
        llm = ChatGroq(temperature=0, groq_api_key=groq_api_key,model_name="meta-llama/llama-4-maverick-17b-128e-instruct")
        ollama_llm = ChatOllama(model="granite-vision-cpu:latest")

        # Configure tools
        configure_tools(vectorstore=vs, embedding_service=embedding_service, llm=llm)

        # Build the graph
        graph_builder = Graph(vectorstore=vs, embedding_service=embedding_service, llm=llm)
        runnable = graph_builder.build_graph()

        # 4) Visualize the graph
        try:
            with open("graph.png", "wb") as f:
                f.write(runnable.get_graph().draw_mermaid_png())
            print("✅ Graph visualization saved as graph.png")
        except Exception as e:
            print(f"⚠️ Could not save graph visualization: {e}")


        # 3) Retrieve with TEXT query
        print("\n🔍 TEXT QUERY: 'what revenue trends across Q1, Q2, and Q3. and what the type of visual elements are present in the document and the colors of q1 and q2 and q3'")
        results_with_scores = vs.retrieve_by_query_with_scores("what revenue trends across Q1, Q2, and Q3. and what the type of visual elements are present in the document and the colors of q1 and q2 and q3?",embedding_service,k=6)
        for doc, score in results_with_scores:
            print({
                "score": float(score),
                "source": doc.metadata.get("source"),
                "type": doc.metadata.get("type"),
                "snippet": (doc.page_content or "")
            })

        # 4) Retrieve with IMAGE query
        print("\n\n🖼️ IMAGE QUERY: 'data/image1.png'")
        image_results = vs.retrieve_by_image_with_scores("data/image1.png", embedding_service, k=3)
        for doc, score in image_results:
            print({
                "score": float(score),
                "source": doc.metadata.get("source"),
                "type": doc.metadata.get("type"),
                "snippet": (doc.page_content or "")
            })


        # 5) Run a test query
        query = "what is the CAD in synera process stands for?"
        query1 = "what revenue trends across Q1, Q2, and Q3. and what the type of visual elements are present in the document and the colors of q1 and q2 and q3?"
        query2 = "what is DFER"
        #initial_state = {"messages": [HumanMessage(content=query)]}
        #final_state = runnable.invoke(initial_state)

        #print("\n\n📝 Final Response:")
        #print(final_state)


        # --- Test Case 1: Best Match Mode (Default) ---
        print("\n\n" + "="*50)
        print("🚀 STARTING TEST: BEST MATCH MODE")
        print("="*50 + "\n")
        configure_retriever_mode(use_strict_context=False)
        for chunk in runnable.stream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": query2,
                    }
                ],
                "turns": 0
            },
        ):
            for node, update in chunk.items():
                print("Update from node", node)
                if "messages" in update:
                    update["messages"][-1].pretty_print()
                print("\n\n")


        # --- Test Case 2: Strict Context Mode ---

        print("\n\n" + "="*50)
        print("🚀 STARTING TEST: STRICT CONTEXT MODE")
        print("="*50 + "\n")
        configure_retriever_mode(use_strict_context=True)
        for chunk in runnable.stream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": query1,
                    }
                ],
                "turns": 0
            },
        ):
            for node, update in chunk.items():
                print("Update from node", node)
                if "messages" in update:
                    update["messages"][-1].pretty_print()
                print("\n\n")


        # The final answer is in the last message of the stream
        #final_state = runnable.invoke({"messages": [{"role": "user","content": query}]})
        #print("================================ Final Answer ================================")
        #final_state['messages'][-1].pretty_print()
        #print("==============================================================================")
    else:
        print("Invalid mode selected. Please enter 'doc', 'doc10', or 'pipe'.")

    # Multilagnual embedding model.


