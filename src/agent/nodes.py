# src/agent/nodes.py

import logging
from typing import Dict, Any, List
from pathlib import Path
import json
import requests
import re
import time
from urllib.parse import urlparse
import hashlib
from PIL import Image
import base64
import io

# Framework Imports
from src.agent.state import AgentState
from src.safety.risk_classifier import RiskClassifier
from src.safety.guardrails import Guardrails
from src.rag.vector_store import MedicalVectorStore
from src.tools.web_scraper import MedicalWebScraper
from config.settings import *
from src.tools.helpers import normalize_query, detect_search_domain, build_metadata
from src.tools.system_prompt import SYSTEM_BEHAVIORAL_SPEC
from src.preprocessing.image_processor import MedicalImageProcessor

# Configure logging
logger = logging.getLogger("medigemma_nodes")

# --- Initialize Core Components Once (Saves Memory & Inits Database) ---
risk_classifier = RiskClassifier()
guardrails = Guardrails()
vector_store = MedicalVectorStore()
web_scraper = MedicalWebScraper() 
image_processor = MedicalImageProcessor() 


# =======================================================================
# 🌐 PIPELINE EXECUTION NODES INDEPENDENT IMPLEMENTATION
# =======================================================================

def run_dialogue_manager(state: AgentState) -> Dict[str, Any]:
    """
    Node 1: Hardened Dialogue Manager FSM.
    Implements strict decision priority ordering with decoupled topic taxonomy
    (medical/nutrition) and retrieval intent detection.
    """
    logger.info("=== [Node: Dialogue Manager FSM] ===")
    
    user_query = state.get("user_query", "").strip()
    image_path = state.get("image_path")
    query_lower = user_query.lower()
    
    previous_state = state.get("dialog_state", "chat").lower().strip()
    existing_subject = state.get("clinical_subject")
    followup_pending = state.get("followup_pending", False)
    trace = state.get("routing_trace", []) or []

    # -------------------------------------------------------------
    # 🎯 DECISION STEP 1: Safety Blocked?
    # -------------------------------------------------------------
    is_safe, output_text = guardrails.validate_input(user_query)
    if not is_safe:
        return {
            "is_safe": False,
            "dialog_state": "blocked",
            "requires_retrieval": False,
            "agent_response": output_text,
            "routing_trace": trace + ["dialogue_mgr -> blocked_refusal"]
        }

    # Extract taxonomy concepts & retrieval intent independently
    has_medical_concepts = any(
        re.search(rf"\b{re.escape(k)}\b", query_lower)
        for k in guardrails.MEDICAL_HEALTH_KEYWORDS
    )
    is_nutrition_query = any(
        re.search(rf"\b{re.escape(k)}\b", query_lower)
        for k in NUTRITION_KEYWORDS
    )
    has_retrieval_intent = any(phrase in query_lower for phrase in RETRIEVAL_PHRASES)

    # Dynamic Clinical Subject Extraction / State Preservation
    clinical_subject = existing_subject
    for pattern in MEDICAL_SUBJECT_PATTERNS:
        match = re.search(pattern, query_lower)
        if match:
            clinical_subject = match.group(1)
            break

    current_state = previous_state
    requires_retrieval = False
    is_affirmation_response = False

    # -------------------------------------------------------------
    # 🎯 DECISION STEP 2: Explicit Retrieval Request?
    # -------------------------------------------------------------
    if has_retrieval_intent:
        current_state = "clinical"
        requires_retrieval = True
        followup_pending = False
        logger.info("🔍 Explicit retrieval request detected. Setting dialog_state=clinical, requires_retrieval=True.")

    # -------------------------------------------------------------
    # 🎯 DECISION STEP 3: Image Present + First Turn?
    # -------------------------------------------------------------
    elif image_path and previous_state in ["chat", "idle"]:
        current_state = "multimodal_triage"
        requires_retrieval = False
        followup_pending = True  # Enable follow-up tracking for Turn 2
        logger.info("📸 Image asset detected on Turn 1. Setting dialog_state=multimodal_triage, requires_retrieval=False.")

    # -------------------------------------------------------------
    # 🎯 DECISION STEP 4: Pending "Yes" / Affirmation Response?
    # -------------------------------------------------------------
    elif followup_pending and query_lower in AFFIRMATION_EXPRESSIONS:
        current_state = "clinical"
        requires_retrieval = False
        is_affirmation_response = True
        followup_pending = False
        logger.info("💬 User affirmation ('Yes') received. Prompting for explicit topic selection.")

    # -------------------------------------------------------------
    # 🎯 DECISION STEP 5: Clinical or Nutrition Conversation
    # -------------------------------------------------------------
    elif (
        has_medical_concepts 
        or is_nutrition_query 
        or previous_state == "clinical"
    ):
        current_state = "clinical"
        requires_retrieval = has_retrieval_intent
        followup_pending = False
        logger.info(
            f"🩺 Clinical/nutrition conversation turn. "
            f"nutrition={is_nutrition_query}, retrieval={requires_retrieval}"
        )

    # -------------------------------------------------------------
    # 🎯 DECISION STEP 6: General Casual Chat
    # -------------------------------------------------------------
    else:
        current_state = "chat"
        requires_retrieval = False
        followup_pending = False

    # Restrict image reprocessing strictly to Turn 1 multimodal triage
    process_image = (
        image_path is not None 
        and current_state == "multimodal_triage"
    )

    # Formulate instant response if Turn 2 is a generic affirmation ("Yes")
    agent_response = ""
    if is_affirmation_response:
        agent_response = (
            "I'd be glad to provide more details. Which would you like to explore?\n\n"
            "• **Common causes and triggers**\n"
            "• **Treatment and self-care options**\n"
            "• **Warning signs that need urgent attention**"
        )

    return {
        "is_safe": True,
        "dialog_state": current_state,
        "requires_retrieval": requires_retrieval,
        "clinical_subject": clinical_subject,
        "followup_pending": followup_pending,
        "agent_response": agent_response,
        "user_query": user_query,
        "image_path": image_path if process_image else None,
        "process_image": process_image,
        "routing_trace": trace + [
            f"dialogue_mgr: state={current_state} | requires_retrieval={requires_retrieval} | "
            f"nutrition={is_nutrition_query} | subject={clinical_subject}"
        ]
    }

def run_risk_classification(state: AgentState) -> Dict[str, Any]:
    """Node 2: Triage risk classification."""
    logger.info("=== [Node: Risk Classification] ===")
    user_query = state.get("user_query", "")
    classification = risk_classifier.classify_query(user_query)
    
    return {
        "risk_level": classification.get("risk_level", "LOW"),
        "risk_metadata": classification
    }


def run_multimodal_processing(state: AgentState) -> Dict[str, Any]:
    """Node 3: Direct Base64 image payload preparation for Turn 1 visual triage."""
    logger.info("=== [Node: Multimodal Processing] ===")
    
    # Strictly check process_image authorization flag from Dialogue Manager
    if not state.get("process_image", False):
        logger.info("⏩ Skipping image preprocessing (Turn 2+ or text-only turn).")
        return {"processed_image_payload": None}

    image_path = state.get("image_path")
    if not image_path or not Path(image_path).exists():
        logger.warning(f"⚠️ Target file path missing: {image_path}")
        return {"image_path": None, "processed_image_payload": None}
        
    try:
        proc_result = image_processor.process(image_path)
        return {
            "image_path": image_path,
            "processed_image_payload": proc_result.image_base64
        }
    except Exception as e:
        logger.error(f"❌ Image Preprocessing Node failure: {e}")
        return {"processed_image_payload": None}


def run_vector_search(state: AgentState) -> Dict[str, Any]:
    """
    Node 4: Contextual Vector Search.
    Executes ONLY when requires_retrieval == True and resolves query using clinical_subject.
    """
    logger.info("=== [Node: Local Vector Retrieval] ===")
    start_time = time.time()
    
    if not state.get("requires_retrieval", False):
        logger.info("ℹ️ requires_retrieval is False. Bypassing vector retrieval.")
        return {
            "retrieved_context": [],
            "context_sources": [],
            "top_retrieval_distance": 0.0,
            "retrieval_query": None
        }

    user_query = state.get("user_query", "").strip()
    clinical_subject = state.get("clinical_subject", "")

    # Contextual Query Resolution: "How is this treated?" -> "urticaria How is this treated?"
    search_query = f"{clinical_subject} {user_query}".strip() if clinical_subject else user_query
    
    search_results = []
    top_distance = 1.0
    
    try:
        logger.info(f"🔍 Executing Context-Resolved Vector Search: '{search_query}'")
        search_results = vector_store.hybrid_search(query=search_query, k=3, score_threshold=0.55)
        if search_results:
            top_distance = float(search_results[0].metadata.get("score", 0.35))
    except Exception as e:
        logger.error(f"❌ Vector search failed: {e}")

    retrieved_chunks = [doc.page_content for doc in search_results]
    sources = list(set(doc.metadata.get("source", "verified_doc") for doc in search_results))

    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(f"⏱️ Retrieval Complete. Latency: {elapsed_ms:.2f} ms | Chunks: {len(retrieved_chunks)}")
    
    return {
        "retrieved_context": retrieved_chunks,
        "context_sources": sources,
        "retrieval_query": search_query,
        "top_retrieval_distance": top_distance
    }


def run_web_search_tool(state: AgentState) -> Dict[str, Any]:
    """
    Node 5: Trusted Web Search Fallback Engine.

    Web retrieval is strictly intent-driven. The Dialogue Manager determines whether
    retrieval is required; this node executes trusted web retrieval only when 
    `requires_retrieval` is True (e.g., when local vector retrieval fails or yields insufficient context).
    """
    logger.info("=== [Node: Live Web Search Fallback] ===")
    start_time = time.time()

    user_query = state.get("user_query", "").strip()
    requires_retrieval = state.get("requires_retrieval", False)
    clinical_subject = state.get("clinical_subject")
    active_graph_risk = state.get("risk_level") or "LOW"

    # 1. Intent Gate: Skip web retrieval if retrieval was not authorized by Dialogue Manager
    if not requires_retrieval:
        logger.info("⏭️ Web retrieval skipped: requires_retrieval=False.")
        return {
            "retrieved_context": [],
            "context_sources": [],
            "retrieval_query": None,
            "risk_level": active_graph_risk,
            "show_risk_badge": False
        }

    # 2. Context-Aware Query Construction
    clean_query = normalize_query(user_query)
    
    if clinical_subject and clean_query:
        search_query = f"{clinical_subject} {clean_query}".strip()
    else:
        search_query = clean_query

    logger.info(f"🔍 Context-resolved web search query: '{search_query}'")

    # 3. Domain Intent Detection & Search Target Scoping
    search_domain = detect_search_domain(search_query)

    if search_domain == "nutrition":
        logger.info("🥗 Scoping search intent execution to nutrition domains.")
        search_target = f"{search_query} diet nutrition"
    else:
        logger.info("🩺 Scoping search intent execution to clinical domains.")
        search_target = search_query

    # Fetch whitelisted target URLs
    target_urls = web_scraper.fetch_whitelisted_urls(
        query=search_target, 
        max_results=4, 
        domain_type=search_domain
    ) or []

    # 4. Domain Authority Scoring & Prioritization
    def score_url(url: str) -> int:
        domain = urlparse(url).netloc.replace("www.", "")
        for trusted_key, info in TRUSTED_WEB_SOURCES.items():
            if trusted_key in domain:
                return info["score"]
        return 0

    target_urls = sorted(target_urls, key=score_url, reverse=True)

    if not target_urls:
        logger.warning("🌐 No trusted online targets resolved.")
        return {
            "retrieved_context": [],
            "context_sources": [],
            "retrieval_query": search_query,
            "risk_level": active_graph_risk,
            "show_risk_badge": False
        }

    retrieved_context = []
    context_sources = []

    # 5. Scrape, Parse & Cache Execution Loop
    for url in target_urls:
        try:
            # Check vector collection cache to avoid dupe scrapes
            try:
                existing_cache = vector_store.vectorstore.get(where={"url": url})
                if existing_cache and existing_cache.get("ids"):
                    logger.info(f"💾 Cache hit. Loading historical vector layers for: {url}")
                    retrieved_context.extend(existing_cache.get("documents", []))
                    parsed_domain = urlparse(url).netloc.replace("www.", "")
                    matched = next((d for d in TRUSTED_WEB_SOURCES if d in parsed_domain), "Source")
                    s_name = TRUSTED_WEB_SOURCES.get(matched, {"name": "Cached Reference"})["name"]
                    if s_name not in context_sources:
                        context_sources.append(s_name)
                    continue
            except Exception as cache_err:
                logger.warning(f"Cache lookup pass bypassed safely: {cache_err}")

            # Execute web scraping
            web_chunks = web_scraper.scrape_to_structured_markdown(url, query_condition=search_query) or []
            
            # Quality Filter: Purge boilerplate text chunks under 120 characters
            web_chunks = [c for c in web_chunks if len(c.get("text", "") if isinstance(c, dict) else str(c)) > 120]
            if not web_chunks:
                continue

            parsed = urlparse(url)
            domain = parsed.netloc.replace("www.", "")
            matched_domain = next((d for d in TRUSTED_WEB_SOURCES if d in domain), "Trusted Source")
            source_info = TRUSTED_WEB_SOURCES.get(matched_domain, {"name": "Trusted Source", "score": 7})

            path_parts = [p for p in parsed.path.split("/") if p]
            condition = "Diet & Nutrition" if search_domain == "nutrition" else "General Health"
            
            target_triggers = {"conditions", "symptoms", "nutrition"}
            matching_dirs = list(target_triggers.intersection(path_parts))
            if matching_dirs:
                idx = path_parts.index(matching_dirs[0])
                if idx + 1 < len(path_parts):
                    condition = path_parts[idx + 1].replace("-", " ").title()

            processed_texts = []
            processed_metadatas = []
            processed_ids = []

            for chunk_idx, chunk in enumerate(web_chunks[:3]):  # Limit to top 3 quality chunks per page
                chunk_text = chunk["text"] if isinstance(chunk, dict) else str(chunk)
                existing_meta = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
                
                meta = build_metadata(
                    source=source_info["name"],
                    url=url,
                    condition=existing_meta.get("condition", condition),
                    section=existing_meta.get("section_header", "Overview"),
                    trust_score=source_info["score"],
                    risk=active_graph_risk,
                    category="Diet & Nutrition" if search_domain == "nutrition" else "General Medicine"
                )
                
                # Guaranteed Unique Hash ID per Chunk Payload
                unique_seed = f"{url}#{chunk_idx}#{chunk_text}"
                chunk_id = hashlib.sha256(unique_seed.encode("utf-8")).hexdigest()
                
                if chunk_id not in processed_ids:
                    processed_texts.append(chunk_text)
                    processed_metadatas.append(meta)
                    processed_ids.append(chunk_id)

            if processed_texts:
                vector_store.vectorstore.add_texts(
                    texts=processed_texts,
                    metadatas=processed_metadatas,
                    ids=processed_ids
                )
                retrieved_context.extend(processed_texts)
                ref_str = f"{source_info['name']} • {condition}"
                if ref_str not in context_sources:
                    context_sources.append(ref_str)

        except Exception as page_ex:
            logger.exception(f"❌ Isolated failure processing web page: {url} | Reason: {page_ex}")
            continue

    # 6. Cap Maximum Retrieved Web Chunks
    MAX_WEB_CHUNKS = 3
    if len(retrieved_context) > MAX_WEB_CHUNKS:
        logger.info(f"🛑 Capping retrieved web chunks to {MAX_WEB_CHUNKS}. Original count: {len(retrieved_context)}")
        retrieved_context = retrieved_context[:MAX_WEB_CHUNKS]

    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(f"⏱️ Web Scraper Complete. Latency: {elapsed_ms:.2f} ms | Located Chunks: {len(retrieved_context)}")

    return {
        "retrieved_context": retrieved_context,
        "context_sources": context_sources,
        "retrieval_query": search_query,
        "risk_level": active_graph_risk,
        "show_risk_badge": search_domain != "nutrition"
    }
    
def safe_refusal_node(state: AgentState) -> dict:
    """
    Generates tailored, safe refusal responses depending on whether 
    a request was blocked by safety guardrails or simply missing KB context.
    """
    query = state.get("user_query", "")
    is_safe = state.get("is_safe", True)
    risk_level = state.get("risk_level", "low")
    
    # Case 1: Flagged by Safety/Guardrail Policy (Code, Malicious, Non-Medical)
    if not is_safe or risk_level == "BLOCKED":
        refusal_text = (
            "I am a specialized AI medical assistant. I cannot write, debug, "
            "or execute programming code, technical scripts, or non-medical software tasks.\n\n"
            "Please feel free to ask any health, medical, or clinical questions!"
        )
    
    # Case 2: Legitimate medical query, but missing from local offline KB
    else:
        refusal_text = (
            f"I currently do not have the verified institutional clinical documentation for '{query}' "
            "stored in my offline knowledge base. For your safety, I am restricted from providing "
            "guidance without verified local references."
        )

    # Standard medical disclaimer suffix
    disclaimer = (
        "\n\n*Disclaimer: I am an AI medical assistant for educational and informational "
        "purposes only. Always consult a qualified healthcare provider for medical emergencies.*"
    )

    return {
        "agent_response": refusal_text + disclaimer,
        "context_sources": ["System Safety Exclusion Guard"] if not is_safe else ["System Offline Protection Layer"]
    }
    

def run_qwen_generation(state: AgentState) -> Dict[str, Any]:
    """
    Node 7: Context & Skill-Aware MediQwen Response Generation.
    Delegates detailed behavioral logic to Skills specification.
    """
    logger.info("=== [Node: MediQwen Response Generation] ===")

    # -------------------------------------------------------------
    # 0. Short-circuit direct Dialogue Manager responses
    # -------------------------------------------------------------
    if state.get("agent_response"):
        return {
            "agent_response": state["agent_response"],
            "clinical_subject": state.get("clinical_subject"),
            "dialog_state": state.get("dialog_state", "chat"),
            "requires_retrieval": state.get("requires_retrieval", False),
        }

    # -------------------------------------------------------------
    # Extract state
    # -------------------------------------------------------------
    image_payload = state.get("processed_image_payload")
    clinical_subject = state.get("clinical_subject")
    requires_retrieval = state.get("requires_retrieval", False)
    dialog_state = state.get("dialog_state", "chat")
    risk_tier = state.get("risk_level", "low").upper()
    user_query = state.get("user_query", "").strip()

    subject_str = clinical_subject or "Unspecified"

    # ===================================================================
    # 1. MULTIMODAL TRIAGE
    # ===================================================================
    if image_payload and dialog_state == "multimodal_triage":
        user_content_prompt = f"""[APPLY SKILL 1: TENTATIVE MULTIMODAL TRIAGE]

PATIENT INQUIRY: "{user_query}"

Provide a cautious, tentative visual assessment based strictly on observable features in the uploaded asset. 
NEVER state a confirmed diagnosis. End with 2–3 interactive follow-up choices (e.g., triggers, treatment, or warning signs)."""

    # ===================================================================
    # 2. CASUAL CONVERSATION
    # ===================================================================
    elif dialog_state == "chat":
        user_content_prompt = f"""[CASUAL CHAT]

USER INPUT: "{user_query}"

Respond warmly, naturally, and concisely in 1–2 sentences. Do NOT introduce medical disclaimers or retrieval notes."""

    # ===================================================================
    # 3. EVIDENCE-GROUNDED RESPONSE
    # ===================================================================
    elif requires_retrieval:
        retrieved_chunks = state.get("retrieved_context", [])
        
        if retrieved_chunks:
            context_str = "\n\n".join(
                f"Evidence {i + 1}:\n{chunk}"
                for i, chunk in enumerate(retrieved_chunks)
            )
        else:
            context_str = "NO_APPROVED_CLINICAL_CONTEXT_AVAILABLE"

        user_content_prompt = f"""[APPLY SKILL 2 & SKILL 3: EVIDENCE-GROUNDED CLINICAL RESPONSE]

ACTIVE TRIAGE RISK: {risk_tier}
CLINICAL SUBJECT: {subject_str}

### RETRIEVED CLINICAL CONTEXT
{context_str}

### PATIENT INQUIRY
"{user_query}"

INSTRUCTION: Ground your answer strictly in the context above. If NO_APPROVED_CLINICAL_CONTEXT_AVAILABLE is shown, state exactly:
"I don't have sufficient approved clinical context to provide a reliable answer or recommendation for this specific situation." """

    # ===================================================================
    # 4. GENERAL MEDICAL CONVERSATION
    # ===================================================================
    else:
        user_content_prompt = f"""[APPLY SKILL 3 & SKILL 4: GENERAL MEDICAL DISCUSSION]

ACTIVE TRIAGE RISK: {risk_tier}
CLINICAL SUBJECT: {subject_str}

### PATIENT INQUIRY
"{user_query}"

INSTRUCTION: Provide clear, empathetic guidance based on established conversation context. Do NOT invent clinical guidelines, prescriptions, or dosages."""

    # -------------------------------------------------------------
    # Prepare image payload
    # -------------------------------------------------------------
    raw_payload = state.get("processed_image_payload") or ""

    clean_base64 = (
        raw_payload.split("base64,", 1)[-1]
        if "base64," in raw_payload
        else raw_payload
    )

    payload = {
        "model": "mediqwen:latest",
        "prompt": user_content_prompt,
        "options": {
            "temperature": 0.2,
            "num_ctx": 8192,
        },
        "stream": False,
    }

    if clean_base64 and dialog_state == "multimodal_triage":
        payload["images"] = [clean_base64]

    # -------------------------------------------------------------
    # Execute Ollama generation
    # -------------------------------------------------------------
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=240,
        )

        response.raise_for_status()
        generated_text = response.json().get("response", "").strip()

        # Extract & persist clinical subject from Turn 1 visual assessment
        extracted_subject = clinical_subject

        if dialog_state == "multimodal_triage" and not extracted_subject:
            gen_lower = generated_text.lower()

            for pattern in MEDICAL_SUBJECT_PATTERNS:
                match = re.search(pattern, gen_lower)
                if match:
                    extracted_subject = match.group(1)
                    logger.info(
                        "🏷️ Extracted clinical subject '%s' from Turn 1 visual assessment.",
                        extracted_subject,
                    )
                    break

        return {
            "agent_response": generated_text,
            "clinical_subject": extracted_subject,
        }

    except Exception as err:
        logger.error(f"❌ Generation Error: {err}")
        return {
            "agent_response": (
                "An internal system error occurred. "
                "Please seek appropriate medical assistance if your "
                "symptoms are concerning."
            )
        }