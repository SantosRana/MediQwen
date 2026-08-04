"""
ChromaDB Vector Store for Medical RAG
Handles embedding, storage, and retrieval of medical documents.
"""
import os
import logging
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config.settings import (
    CHROMA_PERSIST_DIR, 
    CHROMA_COLLECTION_NAME, 
    EMBEDDING_MODEL
)
from models.embeddings import initialize_embeddings, normalize_vectors
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
    
    def __init__(self):
        self.persist_dir = CHROMA_PERSIST_DIR
        self.collection_name = CHROMA_COLLECTION_NAME
        self.embedding_model = EMBEDDING_MODEL
        
        logger.info(f"Initializing Persistent ChromaDB client at: {self.persist_dir}")
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        
        # 1. Initialize the base LangChain wrapper
        base_embeddings = initialize_embeddings(self.embedding_model)
        
        # 2. Intercept and inject your normalize_vectors function directly into the wrapper methods
        self._apply_custom_normalization(base_embeddings)
        self.embeddings = base_embeddings
        
        self.vectorstore = None
        # Ensure lazy collection initialization fires immediately upon setup
        self._init_collection()

    def _apply_custom_normalization(self, embeddings_obj):
        """
        Dynamically patches the LangChain embedding object's methods to ensure
        all vector calculations pass through your custom NumPy normalization algorithm.
        """
        original_embed_documents = embeddings_obj.embed_documents
        original_embed_query = embeddings_obj.embed_query

        def custom_embed_documents(texts: List[str]) -> List[List[float]]:
            raw_vecs = original_embed_documents(texts)
            # Apply your working 2D NumPy normalization loop
            norm_vecs = normalize_vectors(np.array(raw_vecs))
            return norm_vecs.tolist()

        def custom_embed_query(text: str) -> List[float]:
            raw_vec = original_embed_query(text)
            # Apply your working 1D NumPy normalization loop
            norm_vec = normalize_vectors(np.array(raw_vec))
            return norm_vec.tolist()

        # Use object.__setattr__ to bypass Pydantic extra-fields validation locking
        object.__setattr__(embeddings_obj, "embed_documents", custom_embed_documents)
        object.__setattr__(embeddings_obj, "embed_query", custom_embed_query)
       
    def _init_collection(self):
        """Initializes the LangChain Chroma wrapper framework layer."""
        try:
            self.vectorstore = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
            logger.info(f"Successfully loaded collection: '{self.collection_name}'")
        except Exception as e:
            logger.error(f"❌ Failed to initialize collection: {e}")
            raise e

    def add_documents(self, formatted_chunks: List[Dict]) -> bool:
        """Adds flattened chunk dictionaries directly to ChromaDB."""
        if not formatted_chunks:
            logger.warning("Empty chunk list received.")
            return False
            
        try:
            texts = [c['content'] for c in formatted_chunks]
            metadatas = [c['metadata'] for c in formatted_chunks]
            ids = [c['id'] for c in formatted_chunks]
            
            self.vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            logger.info(f"Successfully inserted {len(formatted_chunks)} nodes into Vector DB.")
            return True
        except Exception as e:
            logger.error(f"❌ Failed inserting chunks: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = 3, score_threshold: float = 0.6) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results
            score_threshold: Minimum similarity score for valid results
        """
        if not query:
            logger.warning("⚠️ Empty query provided")
            return []
            
        try:
            # Switch to with_score to get the similarity metrics
            results_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
            
            valid_docs = []
            for doc, score in results_with_scores:
                # Lower score in distance metrics usually means closer/better match
                # Adjust threshold based on your embedding model's scale
                logger.info(f"🎯 Chunk match candidate distance score: {score:.4f} for {doc.metadata.get('condition', 'unknown')}")
                
                # BGE large with cosine distance usually yields good matches below 0.5-0.6
                if score <= score_threshold:
                    valid_docs.append(doc)
                    
            logger.info(f"🔍 Validated {len(valid_docs)}/{len(results_with_scores)} chunks above confidence requirements.")
            return valid_docs
        
        except Exception as e:
            logger.error(f"❌ Search error occurred: {e}")
            return []
    
    def hybrid_search(
        self,
        query: str,
        k: int = 3,
        score_threshold: float = 0.6,
        lambda_mult: float = 0.7
    ) -> List[Document]:
        """
        Diverse hybrid search combining similarity with Maximal Marginal Relevance (MMR).
        Safely evaluates Euclidean/Cosine distance scores, dropping bad matches.
        """
        if not query:
            logger.warning("⚠️ Empty query for hybrid search")
            return []
            
        try:
            fetch_k = k * 3
            
            # 1. Fetch scoring candidates first using similarity_search_with_score
            candidates_with_score = self.vectorstore.similarity_search_with_score(query, k=fetch_k)
            
            # 2. Filter out candidates that fail the confidence distance criteria
            # Remember: Chroma L2/Cosine distance scales lower (closer to 0.0) for better matches
            valid_candidates = []
            for doc, score in candidates_with_score:
                # logger.info(f"🎯 Hybrid candidate distance score: {score:.4f} for {doc.metadata.get('condition', 'unknown')}")
                if score <= score_threshold:
                    valid_candidates.append(doc)
            
            if not valid_candidates:
                logger.info("ℹ️ Hybrid Search: Zero candidates cleared the score threshold bounds.")
                return []
                
            # 3. If candidates pass, perform MMR diversity extraction manually on the validated subset
            # We re-run MMR query directly to fetch the target clean 'k' count
            results = self.vectorstore.max_marginal_relevance_search(
                query=query,
                k=min(k, len(valid_candidates)),
                fetch_k=len(valid_candidates),
                lambda_mult=lambda_mult
            )
            
            logger.info(f"🔍 Diverse hybrid search finalized {len(results)} valid chunks.")
            return results
            
        except Exception as e:
            logger.error(f"❌ Hybrid search error: {e}")
            return []
    
    def get_retriever(self, score_threshold: float = 0.6, k: int = 3):
        """
        Generates a calibrated LangChain retriever configured to enforce absolute
        similarity threshold cutoff bounds to prevent hallucinated context streams.
        """
        retriever = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": k,
                "score_threshold": score_threshold  # Filters weak results at the retriever layer
            }
        )
        logger.info(f"✅ Secure safety-capped retriever spawned (threshold={score_threshold}).")
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
            """Returns runtime performance statistics via native client checks."""
            try:
                # Ensure lazy collection initialization fires immediately upon setup
                collection = self.client.get_collection(name=self.collection_name)
                return {
                    "name": self.collection_name,
                    "count": collection.count(),
                    "metadata": collection.metadata or {},
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