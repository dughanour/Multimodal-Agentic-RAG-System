from langchain_postgres import PGVector
from langchain_core.documents import Document
from sqlalchemy import text
from typing import List, Optional, Union, Any, Dict, Tuple
from pathlib import Path
from PIL import Image
import numpy as np


class PostgreSQLVectorDB:
    """Production vector store backed by PostgreSQL + pgvector."""

    def __init__(self, connection_string: str, embeddings):
        """
        Args:
            connection_string: PostgreSQL connection URL from DATABASE_URL env var
            embeddings: NomicEmbeddings adapter instance (from Step 2)
        """
        self.vectorstore = PGVector(
            connection=connection_string,
            embeddings=embeddings,
            collection_name="rag_documents",
        )

    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """Add documents with precomputed embeddings to PostgreSQL."""
        text = [doc.page_content for doc in documents]
        embeddings = [emb.tolist() for emb in embeddings]
        metadatas = [doc.metadata for doc in documents]
        self.vectorstore.add_embeddings(texts=text, embeddings=embeddings, metadatas=metadatas)

    def get_retriever(self):
        """Get the retriever instance."""
        return self.vectorstore.as_retriever()
    
    def retrieve_by_query(self, query: str, embedding_service, k: int = 3):
        """Retrieve documents by query."""
        query_embedding = embedding_service.embed_text(query, task_type="search_query")
        return self.vectorstore.similarity_search_by_vector(embedding=query_embedding, k=k)
    
    def retrieve_by_query_with_scores(self, query: str, embedding_service, k: int = 3, filter: Optional[Dict[str, Any]] = None):
        """Retrieve documents by text query, returning (Document, score) pairs."""
        query_embedding = embedding_service.embed_text(query, task_type="search_query")
        return self.vectorstore.similarity_search_with_score_by_vector(embedding=query_embedding, k=k, filter=filter)
    
    def get_existing_sources(self) -> set:
        """Query PostgreSQL for all unique source values already stored."""
        store = self.vectorstore
        with store._make_sync_session() as session:
            collection = store.get_collection(session)
            if collection is None:
                return set()
            
            result = session.execute(
                text("""
                    SELECT DISTINCT cmetadata->>'source' 
                    FROM langchain_pg_embedding 
                    WHERE collection_id = :cid 
                    AND cmetadata->>'source' IS NOT NULL
                """),
                {"cid":str(collection.uuid)}
            )
            
            return {row[0] for row in result}



