from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List, Optional, Union, Any, Dict, Tuple
from pathlib import Path
from PIL import Image
import numpy as np
import pickle
import os


class VectorStore:
    """Manage vector store functionalities"""
    def __init__(self):
        self.all_docs = []
        # start as empty (0-length) array to avoid np.array() TypeError
        self.embedded_documents = np.empty((0,))
        self.vectorstore : Optional[FAISS] = None
        self.retriever = None    

    def load_precomputed(self, documents: List[Document], embeddings: np.ndarray):
        """
        Load documents and their precomputed embeddings (from the ingestion pipeline).
        """
        self.all_docs = documents or []
        self.embedded_documents = embeddings if embeddings is not None else np.array()


    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """
        Incrementally adds new documents and their embeddings to the in-memory vector store.
        """
        if self.vectorstore is None:
            # If the vector store doesn't exist yet, create it from these new documents.
            self.load_precomputed(documents, embeddings)
            self.create_vectorstore()
        else:
            # If it already exists, add the new documents and embeddings to it.
            # FAISS.add_embeddings expects a list of (text, embedding) tuples.
            text_embeddings = list(zip([doc.page_content for doc in documents], embeddings))
            self.vectorstore.add_embeddings(text_embeddings, metadatas=[doc.metadata for doc in documents])
            
            # Also update our own internal lists for consistency.
            self.all_docs.extend(documents)
            self.embedded_documents = np.concatenate([self.embedded_documents, embeddings], axis=0)


    def create_vectorstore(self):
        """
        Create FAISS vector store from the documents and embeddings held in this instance.
        """
        if not self.all_docs:
            print("⚠️ No documents available to create a vector store.")
            return

        self.vectorstore = FAISS.from_embeddings(
            text_embeddings=[(doc.page_content, emb) for doc, emb in zip(self.all_docs, self.embedded_documents)],
            embedding=None,  # Embeddings are pre-computed
            metadatas=[doc.metadata for doc in self.all_docs]
        )


    def get_retriever(self):
        """
        Get the retriever instance.

        Return:
            Retriever instance
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized. Call create_vectorstore first")
        # The retriever is created on-the-fly from the current vectorstore
        return self.vectorstore.as_retriever()

        
    def retrieve(self, query: str, k: int = 3):
        """
        Unified retrieval using CLIP embeddings for both text and images",
        Retrieve relevant documents for a query

        Args:
            query: Search query
            K: Number of retrieved documents
        
        Return:
            List of relevant documents
        """
        if self.retriever is None:
            raise ValueError("Vectorstore not initialized. Call create_vectorstore first")
        return self.retriever.invoke(query)    


    def retrieve_by_query(self, query: str, embedding_service, k: int = 3):
        """
        Retrieve using CLIP-embedded query against the FAISS store built from precomputed embeddings.

        Args:
            query: text query
            embedding_service: instance providing embed_text(query)
            k: number of results
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized. Call create_retriever first")
        query_embedding = embedding_service.embed_text(query, task_type="search_query")
        return self.vectorstore.similarity_search_by_vector(embedding=query_embedding, k=k)


    def retrieve_by_query_with_scores(self, query: str, embedding_service, k: int = 3, filter: Optional[Dict[str, Any]] = None):
        """
        Retrieve using embedded query and return (Document, score) pairs.
        Can be filtered by metadata.

        Args:
            query: text query
            embedding_service: instance providing embed_text(query)
            k: number of results
            filter: Optional dictionary for metadata filtering.
        Returns:
            List[Tuple[Document, float]]
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized. Call create_retriever first")
        query_embedding = embedding_service.embed_text(query, task_type="search_query")
        return self.vectorstore.similarity_search_with_score_by_vector(
            query_embedding, k=k, filter=filter
        )


    def retrieve_by_image(self, image: Union[str, Path, Image.Image], embedding_service, k: int = 3):
        """
        Retrieve using image query against the FAISS store.

        Args:
            image: image path (str/Path) or PIL Image object
            embedding_service: instance providing embed_image(image)
            k: number of results
        Returns:
            List[Document]
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized. Call create_retriever first")
        
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image
        
        # Embed the image
        image_embedding = embedding_service.embed_image(pil_image)
        return self.vectorstore.similarity_search_by_vector(embedding=image_embedding, k=k)


    def retrieve_by_image_with_scores(self, image: Union[str, Path, Image.Image], embedding_service, k: int = 3):
        """
        Retrieve using image query and return (Document, score) pairs.

        Args:
            image: image path (str/Path) or PIL Image object
            embedding_service: instance providing embed_image(image)
            k: number of results
        Returns:
            List[Tuple[Document, float]]
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized. Call create_retriever first")
        
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image
        
        # Embed the image
        image_embedding = embedding_service.embed_image(pil_image)
        return self.vectorstore.similarity_search_with_score_by_vector(
            image_embedding, k=k
        )


    def save_local(self, persist_directory: str):
        """
        Save the current in-memory FAISS index to disk. This will overwrite the existing index.
        """
        os.makedirs(persist_directory, exist_ok=True)
        index_path = os.path.join(persist_directory, "faiss_index")

        if self.vectorstore:
            # Save the current, updated vectorstore directly.
            # This overwrites the old index with the new, combined one.
            self.vectorstore.save_local(index_path)
            # Also save the raw documents and embeddings for potential future use
            with open(os.path.join(persist_directory, "docs.pkl"), "wb") as f:
                pickle.dump(self.all_docs, f)
            with open(os.path.join(persist_directory, "embeddings.npy"), "wb") as f:
                np.save(f, self.embedded_documents)
            print(f"✅ Saved FAISS index and documents to {persist_directory}")
        else:
            print("⚠️ Vectorstore is not initialized, nothing to save.")
    


    def load_local(self, persist_directory: str):
        """
        Load FAISS index, documents, and metadata from disk.
        """
        index_path = os.path.join(persist_directory, "faiss_index")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"No saved FAISS index found at {index_path}")
        
        # Load FAISS index
        self.vectorstore = FAISS.load_local(index_path, embeddings=None, allow_dangerous_deserialization=True)
        self.retriever = self.vectorstore.as_retriever()

        # Load docs and embeddings
        with open(os.path.join(persist_directory, "docs.pkl"), "rb") as f:
            self.all_docs = pickle.load(f)
        self.embedded_documents = np.load(os.path.join(persist_directory, "embeddings.npy"), allow_pickle=True)

        print(f"✅ Loaded FAISS index and documents from {persist_directory}")
