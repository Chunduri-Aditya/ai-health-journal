"""
Local RAG store using Chroma for privacy-first vector storage.
"""

import os
import json
import logging
from typing import List, Dict, Optional
from datetime import datetime

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except (ImportError, Exception) as e:
    CHROMA_AVAILABLE = False
    chromadb = None  # type: ignore
    Settings = None  # type: ignore
    logging.warning(f"Chroma not available. RAG will be disabled. Error: {type(e).__name__}. Install/update with: pip install -U chromadb")


class RAGStore:
    """Local vector store for journal entries and insights."""
    
    def __init__(self, enabled: bool = True, persist_dir: str = "./rag_store"):
        """
        Initialize RAG store.
        
        Args:
            enabled: Whether RAG is enabled
            persist_dir: Directory to persist Chroma data
        """
        self.enabled = enabled and CHROMA_AVAILABLE
        self.persist_dir = persist_dir
        self.client = None
        self.collection = None
        
        if self.enabled:
            try:
                self.client = chromadb.PersistentClient(
                    path=persist_dir,
                    settings=Settings(anonymized_telemetry=False)
                )
                self.collection = self.client.get_or_create_collection(
                    name="journal_entries",
                    metadata={"description": "Journal entries and insights"}
                )
                logging.info(f"RAG store initialized at {persist_dir}")
            except Exception as e:
                logging.error(f"Failed to initialize RAG store: {e}")
                self.enabled = False
        else:
            if not CHROMA_AVAILABLE:
                logging.warning("Chroma not installed. RAG disabled.")
            else:
                logging.info("RAG disabled by configuration")
    
    def add_entry(self, entry: str, insight: str, metadata: Optional[Dict] = None):
        """
        Add a journal entry and insight to the store.
        
        Args:
            entry: Journal entry text
            insight: Generated insight/analysis
            metadata: Optional metadata (e.g., timestamp, session_id)
        """
        if not self.enabled or not self.collection:
            return
        
        try:
            # Combine entry and insight for retrieval
            document = f"ENTRY: {entry}\n\nINSIGHT: {insight}"
            
            # Use entry as ID (or generate unique ID)
            doc_id = f"entry_{datetime.now().isoformat()}"
            
            # Prepare metadata
            meta = metadata or {}
            meta["timestamp"] = datetime.now().isoformat()
            meta["entry_length"] = len(entry)
            
            # Add to collection (Chroma will handle embeddings)
            self.collection.add(
                documents=[document],
                ids=[doc_id],
                metadatas=[meta]
            )
            logging.debug(f"Added entry to RAG store: {doc_id}")
        except Exception as e:
            logging.error(f"Failed to add entry to RAG store: {e}")
    
    def retrieve(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: Search query (journal entry text)
            top_k: Number of results to return
            
        Returns:
            Concatenated retrieved context string
        """
        if not self.enabled or not self.collection:
            return ""
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, 10)  # Cap at 10
            )
            
            if not results or not results.get("documents") or not results["documents"][0]:
                return ""
            
            # Combine retrieved documents
            retrieved_docs = results["documents"][0]
            context_parts = []
            
            for i, doc in enumerate(retrieved_docs):
                context_parts.append(f"[Retrieved Context {i+1}]\n{doc}\n")
            
            return "\n".join(context_parts)
        
        except Exception as e:
            logging.error(f"Failed to retrieve from RAG store: {e}")
            return ""
    
    def clear(self):
        """Clear all entries from the store."""
        if not self.enabled or not self.client:
            return
        
        try:
            self.client.delete_collection(name="journal_entries")
            self.collection = self.client.get_or_create_collection(
                name="journal_entries",
                metadata={"description": "Journal entries and insights"}
            )
            logging.info("RAG store cleared")
        except Exception as e:
            logging.error(f"Failed to clear RAG store: {e}")


# Global RAG store instance
_rag_store: Optional[RAGStore] = None


def get_rag_store(enabled: bool = True) -> RAGStore:
    """Get or create global RAG store instance."""
    global _rag_store
    if _rag_store is None:
        _rag_store = RAGStore(enabled=enabled)
    return _rag_store
