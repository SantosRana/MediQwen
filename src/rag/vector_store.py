"""
ChromaDB Vector Store for Medical RAG
Handles embedding, storage, and retrieval of medical documents.
"""
import os
import logging
from typing import List, Dict, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config.settings import (
    CHROMA_PERSIST_DIR, 
    CHROMA_COLLECTION_NAME, 
    EMBEDDING_MODEL
)
from models.embeddings import initialize_embeddings
from typing import Dict, Iterable, List, Union, Sequence
from dataclasses import asdict
from rag.knowledge_base import MedicalChunk

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def transform_kb_chunks_to_docs(kb_chunks: List[MedicalChunk]) -> List[Dict]:
    """
    Transform native MedicalChunk dataclass objects directly 
    into the dictionary format expected by the vector store.
    """
    docs = []
    for chunk in kb_chunks:
        # Convert the dataclass instance to Python dictionary
        chunk_dict = asdict(chunk)
        
        docs.append({
            'id': chunk_dict['chunk_id'],
            'content': chunk_dict['text'],
            'metadata': chunk_dict['metadata']
        })
    return docs


class MedicalVectorStore:
    """
    Manages the ChromaDB vector store for medical knowledge retrieval.
    Uses BGE embeddings for high-quality medical semantic search.
    """
    
    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL
    ):
        """Initialize the vector store."""
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Initialize embeddings via centralized factory
        logger.info(f"🔧 Loading embeddings: {embedding_model}")
        try:
            self.embeddings = initialize_embeddings(model_name=embedding_model)
            logger.info(f"✅ Embeddings initialized: {embedding_model}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize embeddings: {e}")
            raise
            
        # Initialize ChromaDB client with proper settings
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        )
        
        # Lazy-load the collection on first access for memory efficiency
        self._collection = None
        
    def _init_collection(self):
        """
        Initialize or load the collection safely.
        Uses client.get_or_create_collection to prevent accidental data loss.
        """
        try:
            collection_obj = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": f"Medical KB Collection: {self.collection_name}"}
            )
            
            # Wrap the raw collection in LangChain Chroma wrapper for consistent interface
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                client=self.client,
                persist_directory=str(self.persist_dir)
            )
            
            count = collection_obj.count()
            
            if count == 0:
                logger.info(f"➕ Initialized new collection '{self.collection_name}' (Empty)")
            else:
                logger.info(f"📦 Loaded existing collection '{self.collection_name}' with {count} documents")
                
            return count
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize collection: {e}")
            raise
    
    @property 
    def simulated_collection(self):
        """Simulate a .collection attribute for backward compatibility (returns proxy object)."""
        try:
            raw_coll = self.client.get_or_create_collection(name=self.collection_name)
            return type('CollectionProxy', (), {'count': lambda s=raw_coll: s.count()})()
        except Exception as e:
            logger.error(f"❌ Error getting simulated collection count: {e}")
            # Return a dummy object with 0 count if error occurs
            class DummyCount:
                def __init__(self, c): self.__dict__['count'] = lambda: c
            return DummyCount(0)

    def _validate_document(self, doc_dict: Dict) -> Optional[Dict]:
        """Validate and sanitize a single document before adding."""
        if not doc_dict:
            return None
            
        content = doc_dict.get('content', '').strip()
        if not content:
            logger.warning(f"⚠️ Skipping empty document: {doc_dict.get('id', 'unknown')}")
            return None
        
        metadata = doc_dict.get('metadata', {})
        if not isinstance(metadata, dict):
            metadata = {'source': 'unknown'}
            
        return {
            'id': doc_dict.get('id', Path(doc_dict.get('file_path', 'unknown')).stem),
            'content': content,
            'metadata': metadata
        }
    
    def add_documents(self, documents: List[Dict], batch_size: int = 100):
        """
        Add documents to the vector store with validation.
        
        Args:
            documents: List of dicts with 'id', 'content', 'metadata'
            batch_size: Number of documents per batch
        """
        if not documents:
            logger.info("ℹ️ No documents to add")
            return 0
            
        logger.info(f"⬆️ Adding {len(documents)} documents to vector store...")
        
        valid_docs = []
        for doc in documents:
            validated_doc = self._validate_document(doc)
            if validated_doc:
                valid_docs.append(Document(
                    page_content=validated_doc['content'],
                    metadata=validated_doc['metadata'],
                    id=validated_doc['id']
                ))
        
        added_count = len(valid_docs)
        skipped_count = len(documents) - added_count
        
        if skipped_count > 0:
            logger.info(f"⚠️ Skipped {skipped_count} invalid documents")
        
        if not valid_docs:
            logger.warning("⚠️ All documents were invalid or empty")
            return 0
            
        # Add in batches
        for i in range(0, len(valid_docs), batch_size):
            batch = valid_docs[i:i+batch_size]
            try:
                self.vectorstore.add_documents(batch)
                batch_num = i // batch_size + 1
                logger.debug(f"   Added batch {batch_num}/{(len(valid_docs)-1)//batch_size + 1}")
            except Exception as e:
                logger.error(f"❌ Error adding batch {i}: {e}")
                continue
                
        logger.info(f"✅ Added {added_count} valid documents total")
        return added_count
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        where: Optional[Dict] = None
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results
            where: Optional metadata filter (e.g., {"source": "emergency_protocol"})
        """
        if not query:
            logger.warning("⚠️ Empty query provided")
            return []
            
        try:
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=where
            )
            logger.debug(f"🔍 Found {len(results)} results for query")
            return results
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []
    
    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        where: Optional[Dict] = None,
        lambda_mult: float = 0.7
    ) -> List[Document]:
        """
        Hybrid search combining semantic similarity with keyword matching.
        Uses MMR (Maximal Marginal Relevance) for diversity.
        
        Args:
            query: Search query
            k: Base number of results
            where: Optional metadata filter
            lambda_mult: Balance between relevance and diversity (0-1)
        """
        if not query:
            logger.warning("⚠️ Empty query for hybrid search")
            return []
            
        try:
            fetch_k = k * 3
            results = self.vectorstore.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=where
            )
            
            logger.debug(f"🔍 Hybrid search found {len(results)} diverse results")
            return results
        except Exception as e:
            logger.error(f"❌ Hybrid search error: {e}")
            return []
    
    def get_retriever(self, search_kwargs: Optional[Dict] = None):
        """
        Get a LangChain retriever for use in RAG pipelines.
        """
        if search_kwargs is None:
            search_kwargs = {"k": 5}
            
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs
        )
        
        logger.info("✅ Retriever created with MMR search")
        return retriever
    
    def delete_collection(self):
        """Safely delete the entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"🗑️ Deleted collection '{self.collection_name}'")
            self._init_collection()
        except Exception as e:
            logger.error(f"❌ Error deleting collection: {e}")
            raise
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        try:
            if not hasattr(self, 'vectorstore') or self.vectorstore is None:
                raw_coll = self.client.get_or_create_collection(name=self.collection_name)
                return {
                    "name": self.collection_name,
                    "count": raw_coll.count(),
                    "metadata": raw_coll.metadata,
                    "embedding_model": self.embedding_model
                }
            collection = self.vectorstore._collection if hasattr(self.vectorstore, '_collection') else None
            return {
                "name": self.collection_name,
                "count": 0,
                "metadata": {},
                "embedding_model": self.embedding_model
            }
        except Exception as e:
            logger.error(f"❌ Error getting stats: {e}")
            return {
                "name": self.collection_name,
                "count": 0,
                "metadata": {},
                "embedding_model": self.embedding_model
            }
    
    def clear_all(self):
        """Clear all documents from the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self._init_collection()
            logger.info("🗑️ Cleared all documents from collection")
            return True
        except Exception as e:
            logger.error(f"❌ Error clearing collection: {e}")
            return False


if __name__ == "__main__":
    try:
        logger.info("🧪 Testing MedicalVectorStore...")
        vs = MedicalVectorStore()
        stats = vs.get_collection_stats()
        logger.info(f"📊 Collection stats: {stats}")
        
        results = vs.similarity_search("hypertension treatment guidelines", k=3)
        logger.info(f"🔍 Test search returned {len(results)} results")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")