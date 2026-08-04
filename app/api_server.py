# app/api_server.py
from datetime import datetime
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

# Import your working compiled graph app
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
async def handle_clinical_query(
    user_query: str = Form(...),
    is_online: str = Form("True"),
    modality: Optional[str] = Form("general"), 
    image_file: Optional[UploadFile] = File(None)
):
    temp_img_path = None
    extracted_text_context = ""
    
    try:
        # 1. Parse the stringified boolean safely from form data
        online_bool = is_online.lower() in ["true", "1", "yes"]

        # 2. Handle Multimodal Upload File Ingestion and Advanced Pre-Processing
        if image_file:
            # Create an air-gapped, isolated temp file on the hosting system disk
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_file.filename)[1]) as temp_buffer:
                shutil.copyfileobj(image_file.file, temp_buffer)
                temp_img_path = temp_buffer.name
            
            # Dynamic Modality Ingestion: Avoid defaulting skin/clinical photos to "document"
            effective_modality = modality if modality and modality != "document" else "general"
            if image_file.filename and any(term in image_file.filename.lower() for term in ["derma", "skin", "rash", "hives", "lesion"]):
                effective_modality = "dermatology"

            # RUNTIME PRE-PROCESSING LOOP:
            proc_result = image_processor.process(temp_img_path, modality=effective_modality)
            
            # If the image contained an uploaded lab chart or text matrix, capture it
            if proc_result.extracted_text:
                extracted_text_context = f"\n[Extracted Medical Record/Chart Text]:\n{proc_result.extracted_text}"

        # 3. Assemble and enrich LangGraph Execution Initial State
        initial_state = {
            "user_query": user_query + extracted_text_context,
            "is_online": online_bool,
            "is_safe": True,
            "dialog_state": "chat",
            "image_path": temp_img_path,
            "retrieved_context": [],
            "context_sources": []
        }
        
        # Invoke the graph natively in a safe python thread context
        output_state = clinical_graph_app.invoke(initial_state)
        show_badge = output_state.get("show_risk_badge", True)
        
        print("DEBUG - Keys in Graph Output State:", output_state.keys())
        
        return {
            "agent_response": output_state.get("agent_response", ""),
            "dialog_state": output_state.get("dialog_state", "chat"),
            "context_sources": output_state.get("context_sources", []),
            "risk_level": output_state.get("risk_level", "low") if show_badge else "unrated/nutrition",
            "risk_badge": show_badge
        }
        
    except Exception as e:
        print(f"❌ Backend Execution Crash: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # 4. Rigorous clean-up to avoid disk memory exhaustion leaks
        if temp_img_path and os.path.exists(temp_img_path):
            try:
                os.unlink(temp_img_path)
            except Exception as cleanup_err:
                print(f"⚠️ Temp cleanup warning: {cleanup_err}")
    
from datetime import datetime, timezone

@server.get("/health")
@server.options("/health")
def health_check():
    """
    Dedicated, ultra-lightweight status check route.
    Gives Streamlit a fast route to ping to verify connection uptime.
    """
    # Uses the updated timezone-aware format to replace the deprecated .utcnow() method
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(server, host="127.0.0.1", port=8000)