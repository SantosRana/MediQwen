import streamlit as st
import requests
import os
import time
from datetime import datetime
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="MediQwen: Clinical Intelligence Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/chat")
HEALTH_URL = os.getenv("HEALTH_URL", "http://127.0.0.1:8000/health")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    .stApp { max-width: 1200px; margin: 0 auto; }

    /* Header banner */
    .hero-banner {
        padding: 22px 26px;
        border-radius: 12px;
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #2563EB 100%);
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(30, 58, 138, 0.2);
    }
    .hero-banner h1 { margin: 0 0 4px 0; font-size: 1.8rem; font-weight: 700; color: #FFFFFF; }
    .hero-banner p { margin: 0; opacity: 0.9; font-size: 0.92rem; }

    /* Info / clinical note banner */
    .clinical-banner {
        padding: 12px 16px;
        border-radius: 8px;
        background-color: #F0F9FF;
        border-left: 4px solid #0284C7;
        margin-bottom: 16px;
        font-size: 0.88rem;
        color: #0369A1;
    }

    /* Emergency banner */
    .emergency-banner {
        padding: 14px 16px;
        border-radius: 8px;
        background-color: #FEF2F2;
        border-left: 4px solid #DC2626;
        margin-bottom: 14px;
        color: #991B1B;
        font-weight: 600;
    }

    /* Risk badges */
    .risk-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 14px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .risk-low { background:#E6F4EA; color:#1E7D34; }
    .risk-medium { background:#FFF4E0; color:#A3650F; }
    .risk-high { background:#FFE8D6; color:#B54708; }
    .risk-emergency { background:#FFE1E1; color:#B00020; }

    /* Status pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-online { background:#E6F4EA; color:#1E7D34; }
    .status-offline { background:#FCE8E6; color:#C5221F; }
    .dot { height: 8px; width: 8px; border-radius: 50%; display: inline-block; }
    .dot-online { background:#1E7D34; }
    .dot-offline { background:#C5221F; }

    .disclaimer-text { font-size: 0.78rem; color: #64748B; line-height: 1.4; }
    .latency-tag { font-size: 0.72rem; color: #94A3B8; margin-top: 4px; }
    </style>
""", unsafe_allow_html=True)

NO_CONTEXT_MARKER = "Conversational Skill Matrix"


# --- Backend Status Helper ---
def check_backend_status():
    try:
        res = requests.get(HEALTH_URL, timeout=2)
        return res.status_code == 200
    except Exception:
        return False


def render_risk_and_sources(sources, risk_level, show_emergency_banner=False, is_medical=False, dialog_state="chat"):
    """Shared renderer for risk badge + reference context expander."""
    if NO_CONTEXT_MARKER in sources:
        return
        
    # Standardize string states
    normalized_dialog = str(dialog_state).lower().strip() if dialog_state else "chat"
    normalized_risk = str(risk_level).lower().strip() if risk_level else ""

    # Do not show context expander if the turn was blocked by guardrails
    if dialog_state == "blocked" or not sources or "System Safety Guardrail" in sources:
        return
    
    # Emergency banners always take precedence
    if show_emergency_banner and normalized_risk == "emergency":
        st.markdown(
            "<div class='emergency-banner'>🚨 CRITICAL EMERGENCY PATTERN DETECTED: "
            "Seek immediate, real-world emergency medical care.</div>",
            unsafe_allow_html=True,
        )

    # Render risk badge ONLY for active clinical/multimodal turns
    should_show_risk = (
        is_medical 
        and normalized_dialog in {"clinical", "multimodal_triage"}
        and normalized_risk not in {"unrated/nutrition", ""}
    )

    if should_show_risk:
        st.markdown(f"<span class='risk-badge risk-{normalized_risk}'>{normalized_risk} risk</span>", unsafe_allow_html=True)

    if sources:
        with st.expander("📄 Viewed Reference Context"):
            for src in sources:
                st.write(f"- {src}")


# --- State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "recent_queries" not in st.session_state:
    st.session_state.recent_queries = []

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

backend_online = check_backend_status()

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=68)
    st.title("MediQwen Dashboard")

    # Backend Connection Indicator
    if backend_online:
        st.markdown(
            "<span class='status-pill status-online'><span class='dot dot-online'></span> Backend Active</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<span class='status-pill status-offline'><span class='dot dot-offline'></span> Server Disconnected</span>",
            unsafe_allow_html=True,
        )
        st.caption("Start your FastAPI backend (`uvicorn app.api_server:app`) to begin.")

    st.divider()

    # RAG Settings
    online_toggle = st.toggle(
        "🌐 Enable Web RAG Fallback",
        value=True,
        help="Allows searching whitelisted sources (NHS, WHO, Mayo Clinic) when local vector storage lacks specific documentation.",
    )

    st.divider()

    # Image Asset Uploader
    st.subheader("📷 Multimodal Imagery")
    uploaded_image = st.file_uploader(
        "Attach clinical photo or report",
        type=["png", "jpg", "jpeg"],
        help="Upload skin lesions, rashes, or printed medical reports for vision analysis.",
        key=f"image_uploader_{st.session_state.uploader_key}",
    )
    if uploaded_image:
        st.image(uploaded_image, caption="Attached Asset Preview", use_container_width=True)

    st.divider()

    # RECENT QUERIES SECTION
    st.subheader("🕒 Recent Queries")
    if st.session_state.recent_queries:
        for idx, q in enumerate(reversed(st.session_state.recent_queries[-5:])):
            label = q if len(q) <= 28 else f"{q[:25]}..."
            if st.button(f"🔍 {label}", key=f"recent_q_{idx}", use_container_width=True, help=q):
                st.session_state.selected_recent = q
                st.rerun()
    else:
        st.caption("No queries logged in this session yet.")

    st.divider()

    with st.expander("ℹ️ System Architecture"):
        st.markdown(
            "<p class='disclaimer-text'><b>MediQwen</b> utilizes a local multimodal agent architecture combining "
            "vector RAG retrieval with unified vision-language execution.</p>",
            unsafe_allow_html=True,
        )

    with st.expander("⚠️ Clinical Safety Disclaimer"):
        st.markdown(
            "<p class='disclaimer-text'>This system is for educational and clinical decision support purposes only. "
            "It is not a substitute for professional clinical diagnosis or emergency intervention.</p>",
            unsafe_allow_html=True,
        )


# --- Main Header ---
st.markdown(
    """
    <div class='hero-banner'>
        <h1>🩺 MediQwen Clinical Intelligence Engine</h1>
        <p>Edge Vision-Language Reasoning & Evidence-Grounded Agentic Pipeline</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='clinical-banner'><strong>Notice:</strong> Outputs are grounded in verified clinical knowledge bases. "
    "Always confirm diagnostic observations with a certified healthcare provider.</div>",
    unsafe_allow_html=True,
)


# --- Quick Prompt Chips ---
if not st.session_state.messages:
    st.write("##### **Suggested Quick Start Queries:**")
    col1, col2, col3 = st.columns(3)

    selected_chip = None
    with col1:
        if st.button("🔬 Analyze uploaded rash image", use_container_width=True):
            selected_chip = "Please analyze the attached image and describe the visual skin characteristics."
    with col2:
        if st.button("🥗 Heart-Healthy Diet Tips", use_container_width=True):
            selected_chip = "What are the core evidence-based guidelines for a heart-healthy Mediterranean diet?"
    with col3:
        if st.button("📋 Explain hypertension guidelines", use_container_width=True):
            selected_chip = "Summarize standard clinical lifestyle recommendations for Stage 1 Hypertension."

    if selected_chip:
        st.session_state.selected_recent = selected_chip
        st.rerun()


# --- Render Conversation History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        sources = msg.get("sources", [])

        if msg["role"] == "assistant":
            render_risk_and_sources(sources, msg.get("risk_level"))

        st.markdown(msg["content"])

        if msg.get("image"):
            st.image(msg["image"], caption="Attached Clinical Visual", width=320)

        if msg.get("timestamp") or msg.get("latency"):
            latency_info = f" • ⚡ {msg['latency']}s" if msg.get("latency") else ""
            st.markdown(f"<div class='latency-tag'>{msg.get('timestamp', '')}{latency_info}</div>", unsafe_allow_html=True)


# --- Handle Input Prompt ---
recent_trigger = st.session_state.pop("selected_recent", None)
chat_placeholder = "Ask a clinical question or describe symptoms..." if backend_online else "Backend offline — start the FastAPI server to chat"
user_query = st.chat_input(chat_placeholder, disabled=not backend_online) or recent_trigger

if user_query:
    start_time = time.time()

    if not st.session_state.recent_queries or st.session_state.recent_queries[-1] != user_query:
        st.session_state.recent_queries.append(user_query)

    user_msg = {
        "role": "user",
        "content": user_query,
        "timestamp": datetime.now().strftime("%H:%M"),
    }

    with st.chat_message("user"):
        st.markdown(user_query)
        if uploaded_image:
            img_obj = Image.open(uploaded_image)
            st.image(img_obj, caption="Attached Image", width=300)
            user_msg["image"] = img_obj

    st.session_state.messages.append(user_msg)

    # Process Assistant Stream
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        with st.spinner("MediQwen is analyzing clinical parameters and vision embeddings..."):
            try:
                form_data = {
                    "user_query": user_query,
                    "is_online": str(online_toggle)
                }

                files = None
                if uploaded_image:
                    uploaded_image.seek(0)
                    files = {
                        "image_file": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)
                    }

                # 🎯 Increased timeout to 300s to match backend limit
                res = requests.post(BACKEND_URL, data=form_data, files=files, timeout=300)
                elapsed_time = round(time.time() - start_time, 2)

                if res.status_code == 200:
                    data = res.json()
                    final_answer = data.get("agent_response", "")
                    sources_found = data.get("context_sources", [])
                    risk_level = data.get("risk_level", "low")

                    render_risk_and_sources(sources_found, risk_level, show_emergency_banner=True)
                    response_placeholder.markdown(final_answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_answer,
                        "sources": sources_found,
                        "risk_level": risk_level,
                        "timestamp": datetime.now().strftime("%H:%M"),
                        "latency": elapsed_time
                    })

                    if uploaded_image:
                        st.session_state.uploader_key += 1

                else:
                    st.error(f"Backend Server Error: {res.text}")
            except Exception as e:
                st.error(f"Cannot connect to MediQwen Backend Server. Ensure backend (`uvicorn app.api_server:app`) is running. Details: {e}")

# --- Clear Session Action ---
st.divider()
col_left, col_clear = st.columns([4, 1])
with col_clear:
    if st.button("🗑️ Clear Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.recent_queries = []
        st.session_state.uploader_key += 1
        st.rerun()