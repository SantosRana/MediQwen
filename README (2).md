# 🏥 MediGemma
### Offline Multimodal AI Health Assistant

> **Disclaimer:** MediGemma provides health guidance and general information only. It is **not a substitute for professional medical diagnosis or treatment**. Always consult a qualified healthcare provider for medical decisions.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [RAG Pipeline](#rag-pipeline)
- [Safety System](#safety-system)
- [Usage](#usage)
- [Configuration](#configuration)
- [Notebooks](#notebooks)
- [Roadmap](#roadmap)

---

## Overview

MediGemma is a **lightweight, offline-capable AI health assistant** built on Gemma 4. It accepts multimodal inputs — text, medical images (X-rays, skin photos, wounds), and optional voice — to provide risk-aware health guidance in environments where internet access and healthcare infrastructure are limited.

**Core principles:**
- 🔒 **Privacy-first** — runs fully offline, no data leaves the device
- ⚠️ **Safety-aware** — every response includes a risk classification and appropriate escalation guidance
- 🌍 **Accessible** — designed for low-resource hardware and non-expert users
- 📚 **Evidence-grounded** — all responses are backed by a curated medical knowledge base via RAG

---

## Problem Statement

Millions of people globally lack immediate access to healthcare services due to:

- Geographic isolation (rural/remote communities)
- Limited medical infrastructure in low-income regions
- Poor or no internet connectivity
- High cost of professional consultation

Existing AI health tools are predominantly cloud-dependent, making them unusable in offline or privacy-sensitive environments. MediGemma bridges this gap.

---

## Architecture

```
User Input (text / image / audio)
           │
           ▼
   ┌───────────────────┐
   │  Input Processor  │  ← image_processor.py, audio_processor.py
   └────────┬──────────┘
            │
            ▼
   ┌───────────────────┐
   │  Safety Guardrail │  ← guardrails.py (pre-check)
   └────────┬──────────┘
            │
            ▼
   ┌───────────────────┐       ┌──────────────────────┐
   │   RAG Retriever   │──────▶│  ChromaDB Vector Store│
   └────────┬──────────┘       └──────────────────────┘
            │  (top-k relevant medical docs)
            ▼
   ┌───────────────────┐
   │   Gemma 4 Model   │  ← gemma_loader.py
   │  (Reasoning Core) │
   └────────┬──────────┘
            │
            ▼
   ┌───────────────────┐
   │  Risk Classifier  │  ← risk_classifier.py
   └────────┬──────────┘
            │
            ▼
   ┌───────────────────┐
   │  Safety Guardrail │  ← guardrails.py (post-check)
   └────────┬──────────┘
            │
            ▼
      Final Response
    (guidance + risk level
      + escalation advice)
```

The agentic workflow is orchestrated via **LangGraph** (`medigemma_graph.py`), enabling conditional routing based on risk level and input type.

---

## Project Structure

```
medigemma/
├── config/
│   └── settings.py              # Global config, constants, model paths
│
├── data/
├── src/
│   ├── models/
│   │   ├── gemma_loader.py      # Gemma 4 multimodal model loader
│   │   └── embeddings.py        # Embedding model (sentence-transformers)
│   │
│   ├── preprocessing/
│   │   ├── image_processor.py   # Image normalization & standardization│   │
│   ├── rag/
│   │   ├── vector_store.py      # ChromaDB initialization & management
│   │   └── knowledge_base.py    # Document ingestion & chunking
│   │
│   ├── safety/
│   │   ├── risk_classifier.py   # Risk level: LOW / MEDIUM / HIGH / EMERGENCY
│   │   └── guardrails.py        # Pre/post-response safety enforcement
│   │
│   ├── agents/
│   │   └── medigemma_graph.py   # LangGraph agentic workflow
│   │
│   └── utils/
│       └── helpers.py           # Shared utilities
│
├── notebooks/
│   ├── 01_setup_knowledge_base.ipynb
│   ├── 02_test_multimodal.ipynb
│   └── 03_run_agentic_rag.ipynb
│
├── app/
│   └── streamlit_interface.py   # Streamlit web UI
│
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- 8GB+ RAM (16GB recommended for Gemma 4)
- CUDA-compatible GPU (optional, but recommended)
- ~10GB disk space for model weights

### 1. Clone the repository

```bash
git clone https://github.com/yourname/medigemma.git
cd medigemma
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Gemma 4 weights

```bash
# Requires a Hugging Face account with Gemma access
huggingface-cli login
python -c "from src.models.gemma_loader import download_model; download_model()"
```

### 5. Build the knowledge base

```bash
# Place your medical documents in data/knowledge_base/
python -m src.rag.knowledge_base --ingest
```

### 6. Launch the app

```bash
streamlit run app/streamlit_interface.py
```

---

## RAG Pipeline

MediGemma uses a **Retrieval-Augmented Generation (RAG)** pipeline to ground responses in verified medical documents, reducing hallucination and improving accuracy.

### Components

| Component | File | Description |
|-----------|------|-------------|
| Vector Store | `rag/vector_store.py` | ChromaDB persistent store with medical embeddings |
| Retriever | `rag/retriever.py` | Semantic similarity search (top-k with MMR reranking) |
| Knowledge Base | `rag/knowledge_base.py` | Document ingestion, chunking, and metadata tagging |
| Embeddings | `models/embeddings.py` | `sentence-transformers/all-MiniLM-L6-v2` (offline) |

### Supported Knowledge Base Formats

- `.txt`, `.md` — plain text medical documents
- `.pdf` — clinical guidelines, WHO protocols
- `.json` — structured symptom/medication databases

### RAG Configuration

```python
# config/settings.py
RAG_CHUNK_SIZE = 512
RAG_CHUNK_OVERLAP = 64
RAG_TOP_K = 5
RAG_MMR_DIVERSITY = 0.3
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = "./data/chroma_db"
```

---

## Safety System

Every response is assigned one of four risk levels:

| Level | Color | Meaning | Action |
|-------|-------|---------|--------|
| 🟢 LOW | Green | Minor, self-manageable | Home care guidance provided |
| 🟡 MEDIUM | Yellow | Warrants monitoring | Schedule a clinic visit |
| 🔴 HIGH | Red | Needs prompt attention | Seek care within 24 hours |
| 🚨 EMERGENCY | Flashing | Life-threatening signs | Call emergency services immediately |

**MediGemma will never:**
- Provide definitive diagnoses
- Recommend specific prescription medications
- Discourage seeking professional care
- Override an EMERGENCY classification

---

## Usage

### Text Query

```python
from src.agents.medigemma_graph import MediGemmaGraph

agent = MediGemmaGraph()
response = agent.run(
    query="I have had a persistent cough and mild fever for 3 days.",
    input_type="text"
)
print(response["guidance"])
print(response["risk_level"])
```

### Image + Text Query

```python
response = agent.run(
    query="What is this rash on my arm?",
    image_path="./images/rash_photo.jpg",
    input_type="multimodal"
)
```

### Streamlit UI

The web interface supports drag-and-drop image upload, voice input (if microphone is available), and displays risk levels with color-coded visual indicators.

---

## Configuration

All global settings are managed in `config/settings.py`:

```python
# Model
GEMMA_MODEL_ID = "google/gemma-4-9b-it"
DEVICE = "cuda"                  # or "cpu", "mps"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.3                # Low temp for factual medical responses

# RAG
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = "./data/chroma_db"
RAG_TOP_K = 5

# Safety
EMERGENCY_KEYWORDS = ["chest pain", "difficulty breathing", "unconscious", ...]
MAX_RISK_OVERRIDE = False        # Never allow overriding EMERGENCY level

# App
APP_TITLE = "MediGemma Health Assistant"
DISCLAIMER_ENABLED = True
```

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_setup_knowledge_base.ipynb` | Ingest and chunk documents, build ChromaDB |
| `02_test_multimodal.ipynb` | Test image + text inputs with Gemma 4 |
| `03_run_agentic_rag.ipynb` | Full end-to-end pipeline walkthrough |

---

## Roadmap

- [x] Offline text-based RAG pipeline
- [x] Multimodal image input (Gemma 4)
- [x] Risk classification system
- [x] LangGraph agentic workflow
- [ ] Audio input via Whisper (offline)
- [ ] Multilingual support (WHO priority languages)
- [ ] Federated knowledge base updates
- [ ] Android/iOS packaging (Termux / CoreML)
- [ ] Clinical validation with medical professionals

---

## Contributing

Contributions are welcome. Please open an issue first to discuss proposed changes. All contributions must respect the safety principles outlined in this README.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

Built with [Gemma 4](https://ai.google.dev/gemma) · [LangGraph](https://langchain-ai.github.io/langgraph/) · [ChromaDB](https://www.trychroma.com/) · [Streamlit](https://streamlit.io/)
