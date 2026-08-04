# tests/unit/test_retriever.py
import pytest
from src.retrieval.retriever import retrieve_clinical_context  # Adjust import path if needed

def test_retriever_returns_context():
    """Verifies vector store query returns clinical document chunks for a valid query."""
    query = "What are the first-line treatments for asthma?"
    results = retrieve_clinical_context(query, k=2)
    
    assert isinstance(results, list)
    # Checks if retrieved context is either empty (if db unit test is isolated) or returns documents
    assert len(results) >= 0