# app/api_server.py
from datetime import datetime, timezone
import os
import sys
import shutil
import tempfile
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Enforce clean workspace paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import compiled graph app & processor
from src.agent.graph import app as clinical_graph_app
from src.preprocessing.image_processor import MedicalImageProcessor

server = FastAPI(title="MediQwen Clinical Backend Engine")
server.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize Multimodal Preprocessor once on startup
image_processor = MedicalImageProcessor()


@server.post("/chat")
def handle_clinical_query(
    user_query: str = Form(...),
    is_online: str = Form("True"),
    modality: Optional[str] = Form("general"), 
    image_file: Optional[UploadFile] = File(None),
    # --- ADDED: Stateful dialogue tracking inputs ---
    clinical_subject: Optional[str] = Form(None),
    dialog_state: Optional[str] = Form("chat"),
    followup_pending: Optional[str] = Form("False")
):
    temp_img_path = None
    extracted_text_context = ""
    
    try:
        # 1. Parse stringified booleans safely from form data
        online_bool = is_online.lower() in ["true", "1", "yes"]
        followup_pending_bool = str(followup_pending).lower() in ["true", "1", "yes"]

        # Clean empty string subjects sent from client forms
        clean_subject = clinical_subject.strip() if clinical_subject and clinical_subject.strip() else None

        # 2. Handle Multimodal Upload File Ingestion and Pre-Processing
        if image_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_file.filename)[1]) as temp_buffer:
                shutil.copyfileobj(image_file.file, temp_buffer)
                temp_img_path = temp_buffer.name
            
            effective_modality = modality if modality and modality != "document" else "general"
            if image_file.filename and any(term in image_file.filename.lower() for term in ["derma", "skin", "rash", "hives", "lesion"]):
                effective_modality = "dermatology"

            # Pre-processing loop
            proc_result = image_processor.process(temp_img_path, modality=effective_modality)
            
            if proc_result.extracted_text:
                extracted_text_context = f"\n[Extracted Medical Record/Chart Text]:\n{proc_result.extracted_text}"

        # 3. Assemble graph state with active conversational context
        initial_state = {
            "user_query": user_query + extracted_text_context,
            "is_online": online_bool,
            "is_safe": True,
            "dialog_state": dialog_state or "chat",
            "clinical_subject": clean_subject,  # Pass active subject into LangGraph
            "followup_pending": followup_pending_bool,
            "image_path": temp_img_path,
            "retrieved_context": [],
            "context_sources": []
        }
        
        # Invoke LangGraph pipeline
        output_state = clinical_graph_app.invoke(initial_state)
        show_badge = output_state.get("show_risk_badge", True)
        
        print("DEBUG - Keys in Graph Output State:", list(output_state.keys()))
        print(f"DEBUG - Active Subject Preserved: '{output_state.get('clinical_subject')}'")
        
        # 4. Return updated dialogue state back to frontend client
        return {
            "agent_response": output_state.get("agent_response", ""),
            "dialog_state": output_state.get("dialog_state", "chat"),
            "clinical_subject": output_state.get("clinical_subject"),  # <--- CRITICAL
            "followup_pending": output_state.get("followup_pending", False),  # <--- CRITICAL
            "context_sources": output_state.get("context_sources", []),
            "risk_level": output_state.get("risk_level", "low") if show_badge else "unrated/nutrition",
            "risk_badge": show_badge
        }
        
    except Exception as e:
        print(f"❌ Backend Execution Crash: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Clean up temporary files
        if temp_img_path and os.path.exists(temp_img_path):
            try:
                os.unlink(temp_img_path)
            except Exception as cleanup_err:
                print(f"⚠️ Temp cleanup warning: {cleanup_err}")


@server.get("/health")
@server.options("/health")
def health_check():
    """Ultra-lightweight status check route."""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(server, host="127.0.0.1", port=8000, timeout_keep_alive=300)