# src/agent/nodes.py
import os
from typing import Dict, Any, Tuple, Optional
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from PIL import Image

# Import existing preprocessing and safety tools
from src.preprocessing.image_processor import MedicalImageProcessor
from src.safety.guardrails import Guardrails
from src.safety.risk_classifier import RiskClassifier, RiskLevel
from src.rag.vector_store import MedicalVectorStore

# Import the local state type
from .state import MedicalAgentState

# -- Multimodal Coordinated Helper (Move from the notebook's top cell to here) --
def coordinate_multimodal_query(user_text: str, image_input: Optional[Any] = None) -> Tuple[str, Dict[str, Any]]:
    """Helper function to run OCR on images if pytesseract is present."""
    augmented_text = user_text
    image_metadata = {}
    
    if image_input is not None:
        print("📸 OCR/Image processing channel activated...")
        processor = MedicalImageProcessor(enhance_contrast=True, run_ocr=True)
        processed_image = processor.process(image_input)
        image_metadata = processed_image.metadata
        
        if processed_image.extracted_text and processed_image.extracted_text.strip():
            print(f"🎯 OCR Text Extracted ({len(processed_image.extracted_text)} chars)")
            augmented_text += f"\n\n[Context extracted from medical image]:\n{processed_image.extracted_text}"
        else:
            print(f"🔍 Falling back to modality detection: {processed_image.modality}")
            augmented_text += f"\n\n[Context: User attached a clinical photographic classified as: {processed_image.modality}]"
            
    return augmented_text, image_metadata


# -- GRAPH NODES --

def run_guardrail_check(state: MedicalAgentState) -> Dict[str, Any]:
    print("🛡️ [Node: Guardrail] Inspecting input safety...")
    guardrails = Guardrails()
    guardrail_result = guardrails.check_input(state["raw_query"])
    if guardrail_result is None:
        return {"is_safe": True, "refusal_message": ""}
    is_safe, refusal_msg = guardrail_result
    return {"is_safe": is_safe, "refusal_message": refusal_msg}

def run_risk_classification(state: MedicalAgentState) -> Dict[str, Any]:
    print("⚠️ [Node: Classifier] Assessing clinical urgency tier...")
    classifier = RiskClassifier()
    # Updated to call classifier.classify() as validated yesterday
    risk_analysis = classifier.classify(state["raw_query"])
    return {"risk_level": risk_analysis.get("risk_level", "low")}

def run_multimodal_processing(state: MedicalAgentState) -> Dict[str, Any]:
    print("📸 [Node: Multimodal] Running OCR and Query Augmentation...")
    user_text = state["raw_query"]
    image_file = state["image_input"]
    final_search_query, img_meta = coordinate_multimodal_query(user_text, image_file)
    return {"augmented_query": final_search_query, "image_metadata": img_meta}

def run_vector_search(state: MedicalAgentState) -> Dict[str, Any]:
    print("📡 [Node: Retriever] Fetching grounded documents from Chroma collection...")
    # NOTE: In deployment, make sure settings.CHROMA_PERSIST_DIR matches this
    vs = MedicalVectorStore()
    search_query = state.get("augmented_query", state["raw_query"])
    # Hybrid search validated yesterday
    docs = vs.similarity_search(query=search_query, k=3)
    return {"retrieved_docs": docs}

def run_gemma_generation(state: MedicalAgentState) -> Dict[str, Any]:
    print("🤖 [Node: Generator] Executing local offline Gemma 4 generation loop...")
    # Emergency routing logic validated yesterday
    if state["risk_level"] == RiskLevel.EMERGENCY.value:
        print("🚨 Severe clinical urgency detected! Generating emergency text bypass.")
        emergency_text = (
            "⚠️ EMERGENCY DETECTED: If you or someone near you is experiencing severe symptoms "
            "like crushing chest pain, sudden numbness, or severe shortness of breath, please stop "
            "using this tool and call emergency services (like 911) immediately."
        )
        return {"generation": emergency_text, "final_output_with_disclaimer": emergency_text}
        
    llm = OllamaLLM(model="gemma4:e2b", base_url="http://localhost:11434", temperature=0.2, num_predict=512)
    guardrails = Guardrails()
    
    # 1. Format the grounding context cleanly
    context_str = ""
    for idx, doc in enumerate(state.get("retrieved_docs", [])):
        source_name = doc.metadata.get("source", "Verified Clinical Guidance")
        context_str += f"\n[Reference Document {idx+1}] Source: {source_name}\n{doc.page_content}\n"
        
    # 2. Re-use the identical specialized clinical prompt template verified yesterday
    system_prompt = f"""You are MediGemma, an expert, clinical-grade AI medical assistant running fully offline.
Your priority is patient safety and rigid adherence to verified medical protocols.

[CRITICAL GROUNDING INSTRUCTIONS]
- Answer the user's inquiry based strictly and exclusively on the medical context documents provided below.
- If the provided reference documents do not contain the answer, explicitly state: "I do not have enough verified clinical source data to confidently answer this question." 
- Never speculate, hallucinate, or extrapolate outside the provided reference text.

[GROUNDING REFERENCE MATERIAL]:
{context_str}

[PATIENT INPUT (Including extracted visual context)]:
{state.get('augmented_query', state['raw_query'])}

Please formulate your safe, grounded response now:"""

    try:
        model_response = llm.invoke(system_prompt)
        final_text = model_response
        if hasattr(guardrails, 'DISCLAIMER_TEXT') and guardrails.DISCLAIMER_TEXT:
            final_text += f"\n\n🛡️ MANDATORY SAFETY NOTICE:\n{guardrails.DISCLAIMER_TEXT}"
        return {"generation": model_response, "final_output_with_disclaimer": final_text}
    except Exception as e:
        error_msg = f"❌ Connection error: Could not communicate with Ollama service. Error: {e}"
        return {"generation": error_msg, "final_output_with_disclaimer": error_msg}