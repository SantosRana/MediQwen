# src/agent/state.py
from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.documents import Document

class MedicalAgentState(TypedDict):
    """The complete context state of the current medical inquiry."""
    
    # Core User Inputs
    raw_query: str
    image_input: Optional[Any]  # Can be a file path string or a PIL Image object
    
    # Extracted Pipeline Variables
    augmented_query: str
    is_safe: bool
    refusal_message: str
    risk_level: str
    image_metadata: Dict[str, Any]
    
    # Retrieval and Inference Outputs
    retrieved_docs: List[Document]
    generation: str
    final_output_with_disclaimer: str