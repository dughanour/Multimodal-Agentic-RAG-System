from fastapi import UploadFile
from src.document_ingestion.document_ingestion_pipeline import DocumentIngestionPipeline
from src.embeddings.embedding_service import EmbeddingService
from src.vectorstore.vectorstore import VectorStore
from .chat_service import  get_chat_service
import os
import aiofiles
import traceback



class DocumentService:
    """
    Service for handling document ingestion, embedding, and updating the vector store.
    """
    def __init__(self):
        # We get the existing instances from the ChatService to ensure we're using
        # the same vectorstore and embedding service as the agent.
        self.chat_service = get_chat_service()
        self.vectorstore: VectorStore = self.chat_service.vectorstore
        self.embedding_service: EmbeddingService = self.chat_service.embedding_service
        self.ingestion_pipeline = DocumentIngestionPipeline()
        self.upload_dir = "data"
        os.makedirs(self.upload_dir, exist_ok=True)

    async def scrape_and_embed_url(self, url: str, strategy: str = "standard"):
        """
        Scrapes a URL, processes its content, and embeds it into the vector store.
        """
        try:
            print(f"Scraping URL: {url} with strategy: {strategy}")
            sources = [url]
            
            new_docs, embeddings = [], []

            if strategy == "summarize":
                llm = self.chat_service.get_llm()
                new_docs, embeddings = self.ingestion_pipeline.process_and_embed_with_summaries(
                    sources = sources,
                    embedding_service = self.embedding_service,
                    llm = llm,
                    strategy = strategy
                )
                
                
            else:  # standard
                new_docs, embeddings = self.ingestion_pipeline.process_and_embed(
                    sources = sources,
                    embedding_service = self.embedding_service,
                    strategy = "standard"
                )
                

            if new_docs and embeddings.any():
                self.vectorstore.add_documents(new_docs, embeddings)
                self.vectorstore.save_local("./embedded_data")
                print(f"✅ Successfully processed and embedded content from {url}")
                return {
                    "message": "URL content processed successfully.",
                    "url": url,
                    "documents_added": len(new_docs)
                }
            else:
                print(f"ℹ️ No new content to process from {url}")
                return {
                    "message": "URL content already processed or no new content found.",
                    "url": url,
                    "documents_added": 0
                }
        except Exception as e:
            print(f"❌ Error scraping and embedding URL {url}: {e}")
            return {"error": str(e)}

    async def process_and_embed_document(self, file: UploadFile, strategy: str = "standard"):
        """
        Saves an uploaded file, processes it, and adds its embeddings to the vector store.
        """
        # 1. Save the uploaded file locally
        file_path = os.path.join(self.upload_dir, file.filename)
        try:
            async with aiofiles.open(file_path, 'wb') as out_file:
                content = await file.read()
                await out_file.write(content)
        except Exception as e:
            # Handle potential file saving errors
            print(f"Error saving file: {e}")
            return {"error": f"Could not save file: {e}"}

        # 2. Process the document and create embeddings based on the chosen strategy
        # For docling strategy: PDF uses docling, PPTX falls back to summarize
        file_ext = file_path.split('.')[-1].lower() if '.' in file_path else ''
        if strategy == "docling" and file_ext == "pptx":
            print(f"📄 Processing and embedding new document: {file_path} using 'summarize' strategy (PPTX fallback from docling).")
        else:
            print(f"📄 Processing and embedding new document: {file_path} using '{strategy}' strategy.")

        new_docs, embeddings = [], []  # Initialize here

        try:
            if strategy in ("summarize", "docling"):
                llm = self.chat_service.get_llm()
                new_docs, embeddings = self.ingestion_pipeline.process_and_embed_with_summaries(
                    sources=[file_path],
                    embedding_service=self.embedding_service,
                    llm=llm,
                    strategy=strategy
                )
            else:  # "standard"
                new_docs, embeddings = self.ingestion_pipeline.process_and_embed(
                    sources=[file_path],
                    embedding_service=self.embedding_service,
                    strategy="standard"
                )
        except Exception as e:
            print(f"❌ Error during document embedding: {e}")
            traceback.print_exc()
            return {"error": str(e)}

        # 3. Add the new documents and embeddings to the existing vector store
        if len(new_docs) > 0:
            print(f"Updating vector store with {len(new_docs)} new document chunks.")
            self.vectorstore.add_documents(new_docs, embeddings)
            self.vectorstore.save_local("./embedded_data")
            print("✅ Vector store updated and saved successfully.")
            return {"message": f"Successfully processed and embedded {file.filename} using '{strategy}' strategy."}
        else:
            print(f"No new content to add from {file.filename}.")
            return {"message": f"No new content to add from {file.filename}"}


# Singleton instance of the DocumentService
document_service = DocumentService()

def get_document_service():
    """Dependency injector for FastAPI."""
    return document_service
