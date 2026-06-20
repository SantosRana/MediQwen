# streamlit_app.py
import os
import sys
import streamlit as st
from PIL import Image

# Ensure project root is in the python path to prevent relative import bugs
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your standardized LangGraph application state machine
from src.agent.graph import app

st.set_page_config(
    page_title="MediGemma — Offline Health Assistant", 
    page_icon="🏥", 
    layout="centered"
)

# Title & Layout Header
st.title("🏥 MediGemma")
st.subheader("Offline Multimodal AI Health Assistant")
st.markdown("---")

# Safety Warning Box
st.warning(
    "⚠️ **Disclaimer:** MediGemma provides general medical protocol guidance only. "
    "It is **not** a substitute for professional clinical diagnosis. "
    "If you are facing a severe health emergency, please contact local emergency services immediately."
)

# Sidebar for Image Inputs & Metadata
with st.sidebar:
    st.header("📎 Input Attachments")
    uploaded_file = st.file_uploader(
        "Upload a medical image or report (X-ray, rash photo, lab document)", 
        type=["jpg", "jpeg", "png", "bmp", "tiff"]
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Attached Clinical Context", use_container_width=True)
    else:
        image = None
        st.info("No files attached. Operating in standard text mode.")

# Maintain persistent conversation states via Streamlit Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat elements
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Collect User Input Query
if user_query := st.chat_input("Ask a clinical protocol or report question..."):
    
    # 1. Surface user query instantly on visual layout
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # 2. Trigger the backend state machine
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        with st.spinner("Processing inquiry through MediGemma State Machine..."):
            # Prepare initialization parameters matching your MedicalAgentState layout
            initial_state = {
                "raw_query": user_query,
                "image_input": image
            }
            
            try:
                # Invoke the compiled graph natively!
                final_agent_result = app.invoke(initial_state)
                response_text = final_agent_result["final_output_with_disclaimer"]
                
                # Render response
                response_placeholder.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
            except Exception as e:
                error_msg = f"❌ Graph Processing Error: {e}"
                response_placeholder.error(error_msg)