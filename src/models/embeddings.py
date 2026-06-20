# src/models/embeddings.py

from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import sys
from pathlib import Path
import numpy as np
from config.settings import (
        EMBEDDING_MODEL as DEFAULT_MODEL,
        DEVICE as DEFAULT_DEVICE
    )

def initialize_embeddings(
    model_name: Optional[str] = None,
    device: Optional[str] = None
) -> HuggingFaceEmbeddings:
    """
    Initialize the HuggingFace embedding model.
    
    Args:
        model_name: Name of the embedding model (default from config).
        device: Device to run on ('cpu', 'cuda', 'auto').
        
    Returns:
        A configured HuggingFaceEmbeddings instance.
    """
    
    # Use provided arguments or fall back to config defaults
    model_name = model_name or DEFAULT_MODEL
    device = device or DEFAULT_DEVICE
    
    print(f"🔍 Initializing embedding model: {model_name}")
    print(f"🔧 Using device: {device}")
    
    # Set device for PyTorch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Configure embeddings
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device} # Normalize vectors for better similarity
    )
    
    return hf_embeddings

def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize embedding vectors to unit length.
    """
    vectors = np.array(vectors)
    if vectors.ndim == 1:  # single vector
        return vectors / np.linalg.norm(vectors)
    elif vectors.ndim == 2:  # batch of vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms
    else:
        raise ValueError("Input must be 1D or 2D array of vectors")

if __name__ == "__main__":
    # Initialize embeddings
    embeddings = initialize_embeddings()
    print("✅ Embeddings initialized successfully!")
    
    # Test encoding a sample sentence
    test_sentence = "Medical symptom: chest pain"
    
    # Document embeddings
    doc_vecs = embeddings.embed_documents([test_sentence])
    doc_vecs = normalize_vectors(np.array(doc_vecs))
    print(f"✅ Sample document embedding shape: {doc_vecs.shape}")
    print(f"✅ Sample document embedding (first 5 values): {doc_vecs[0][:5]}")
    
    # Query embedding
    query_vec = embeddings.embed_query(test_sentence)
    query_vec = normalize_vectors(np.array([query_vec]))[0]
    
    print(f"✅ Sample embedding shape: {query_vec.shape}")
    print(f"✅ Sample embedding (first 5 values): {query_vec[:5]}")
    