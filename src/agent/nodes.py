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
from src.agent.skills_loader import load_skill
from src.agent.state import AgentState
from src.safety.risk_classifier import RiskClassifier
from src.safety.guardrails import Guardrails
from src.safety.scope_classifier import BGEScopeClassifier, classify_scope_with_gate
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
scope_classifier = BGEScopeClassifier()
vector_store = MedicalVectorStore()
web_scraper = MedicalWebScraper() 
image_processor = MedicalImageProcessor() 


# =======================================================================
# 🌐 PIPELINE EXECUTION NODES INDEPENDENT IMPLEMENTATION
# =======================================================================

def run_dialogue_manager(state: AgentState) -> Dict[str, Any]:
    """
    Node 1: Dialogue Manager FSM.
    Enforces safety boundaries, multimodal context boundaries, semantic scope 
    classification, and context resolution.
    Determines authoritative dialogue state, subject memory, and retrieval flags.
    """
    logger.info("=== [Node: Dialogue Manager FSM] ===")

    user_query = state.get("user_query", "").strip()
    
    # 1. Robust image presence detection across both raw path and pre-processed payload
    image_path = state.get("image_path")
    processed_image_payload = state.get("processed_image_payload")
    has_image = bool(image_path or processed_image_payload)

    previous_state = str(state.get("dialog_state", "chat")).lower().strip()
    existing_subject = state.get("clinical_subject")
    
    # Read incoming follow-up pending flag from previous graph state
    followup_pending_in = state.get("followup_pending", False)
    trace = state.get("routing_trace", []) or []
    query_lower = user_query.lower().strip()

    # =============================================================
    # STEP 1 — SAFETY BOUNDARY
    # =============================================================
    is_safe, output_text = guardrails.validate_input(
        user_query,
        has_image=has_image,
    )

    if not is_safe:
        logger.info("🚫 Safety Guardrail triggered. Blocking request.")
        return {
            "is_safe": False,
            "dialog_state": "blocked",
            "scope": "OUT_OF_SCOPE",
            "scope_source": "SAFETY_GUARDRAIL",
            "decision": "BLOCKED",
            "requires_retrieval": False,
            "clinical_subject": existing_subject,
            "followup_pending": followup_pending_in,
            "agent_response": output_text,
            "process_image": False,
            "routing_trace": trace + ["dialogue_mgr -> blocked_refusal"],
        }

    # =============================================================
    # STEP 1.5 — MULTIMODAL EVENT & CONTEXT BOUNDARY EVALUATION
    # =============================================================
    if has_image:
        has_clinical_intent = any(
            phrase in query_lower for phrase in CLINICAL_INFORMATION_INTENTS
        )
        
        # Build dynamic subject continuation matcher using imported CONTINUATION_PATTERNS
        is_explicit_continuation = False
        if existing_subject:
            dynamic_continuation_patterns = CONTINUATION_PATTERNS + [
                r"\b(image|photo|picture)\s+of\s+my\s+" + re.escape(existing_subject) + r"\b"
            ]
            is_explicit_continuation = any(
                re.search(p, query_lower) for p in dynamic_continuation_patterns
            )

        if is_explicit_continuation:
            logger.info("📸 Image attached with explicit continuation -> Preserving subject: '%s'", existing_subject)
            active_subject = existing_subject
        else:
            logger.info("📸 New image detected -> Clearing stale subject: '%s'", existing_subject)
            active_subject = None

        logger.info(
            "🩺 Dialogue State: [multimodal_triage] | Scope: MULTIMODAL | Retrieval: %s | Subject: %s",
            has_clinical_intent,
            active_subject,
        )

        return {
            "is_safe": True,
            "dialog_state": "multimodal_triage",
            "scope": "MULTIMODAL",
            "scope_source": "MULTIMODAL_IMAGE_OVERRIDE",
            "decision": "HIGH_CONFIDENCE",
            "requires_retrieval": has_clinical_intent,
            "clinical_subject": active_subject,
            "followup_pending": True,
            "agent_response": "",
            "user_query": user_query,
            "image_path": image_path,
            "processed_image_payload": processed_image_payload,
            "process_image": True,
            "routing_trace": trace + [
                f"dialogue_mgr: scope=MULTIMODAL | state=multimodal_triage | retrieval={has_clinical_intent} | subject={active_subject}"
            ],
        }

    # =============================================================
    # STEP 2 — SCOPE / TAXONOMY CLASSIFICATION
    # =============================================================
    # Text queries go through BGE Scope Classifier -> Margin Gate -> Context Resolver
    scope_result = classify_scope_with_gate(
        scope_classifier,
        user_query,
        clinical_subject=existing_subject,
    )
    scope = scope_result["scope"]
    scope_source = scope_result["scope_source"]
    scope_debug = scope_result

    # =============================================================
    # STEP 3 — CLINICAL SUBJECT MEMORY / EXTRACTION
    # =============================================================
    clinical_subject = existing_subject
    logger.info(
        "📥 Incoming clinical_subject from state: '%s'",
        existing_subject,
    )

    # -------------------------------------------------------------
    # Explicit topic transition
    # -------------------------------------------------------------
    # A new nutrition query starts a nutrition context rather than
    # inheriting a stale medical condition from the previous turn.
    if scope == "NUTRITION":
        clinical_subject = None
        logger.info("🥗 Nutrition scope detected. Detaching previous clinical_subject context.")

    # -------------------------------------------------------------
    # Medical subject extraction & context preservation
    # -------------------------------------------------------------
    elif scope == "MEDICAL":

        # 1. Configured medical vocabulary (exact/high-confidence patterns)
        matched_new_subject = None
        for pattern in MEDICAL_SUBJECT_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                matched_new_subject = match.group(1).strip()
                logger.info(
                    "🎯 Clinical subject matched from configured pattern: '%s'",
                    matched_new_subject,
                )
                break

        if matched_new_subject:
            clinical_subject = matched_new_subject

        # 2. Dynamic intent-based extraction when no static pattern matched
        elif not clinical_subject:
            for pattern in DYNAMIC_SUBJECT_PATTERNS:
                match = re.search(pattern, query_lower)
                if match:
                    extracted_raw = match.group(1).strip()
                    clean_subject = re.sub(
                        r"^\s*(?:a|an|the|this|that|my|some)\s+",
                        "",
                        extracted_raw,
                        flags=re.IGNORECASE,
                    ).strip()

                    if clean_subject:
                        clinical_subject = clean_subject
                        logger.info(
                            "🎯 Clinical subject extracted via dynamic fallback: '%s'",
                            clinical_subject,
                        )
                        break
        else:
            logger.info("🔄 Preserving existing clinical_subject for follow-up query: '%s'", clinical_subject)
            
    # =============================================================
    # STEP 4 — RETRIEVAL INTENT
    # =============================================================
    has_retrieval_intent = any(
        phrase in query_lower for phrase in CLINICAL_INFORMATION_INTENTS
    )

    # =============================================================
    # STEP 5 — FSM ROUTING & STATE DETERMINATION
    # =============================================================
    current_state = "chat"
    followup_pending_out = False
    is_affirmation_response = False
    agent_response = ""

    # 1. OUT_OF_SCOPE always wins and blocks immediately
    if scope == "OUT_OF_SCOPE":
        current_state = "blocked"
        agent_response = (
            "I am a specialized AI medical assistant. "
            "I can assist with medical, health, wellness, "
            "and nutrition-related questions."
        )

    # 2. Image follow-up affirmation menu response
    elif followup_pending_in and query_lower in AFFIRMATION_EXPRESSIONS:
        current_state = "clinical"
        followup_pending_out = False
        is_affirmation_response = True

        agent_response = (
            "I'd be glad to provide more details. "
            "Which would you like to explore?\n\n"
            "• **Common causes and triggers**\n"
            "• **Treatment and self-care options**\n"
            "• **Warning signs that need urgent attention**"
        )

    # 3. Medical / Nutrition Scope
    elif scope in {"MEDICAL", "NUTRITION"}:
        current_state = "clinical"

    # 4. Explicit retrieval intent
    elif has_retrieval_intent:
        current_state = "clinical"

    # 5. Continue existing clinical conversation
    elif previous_state == "clinical" and existing_subject:
        current_state = "clinical"

    # 6. Default casual conversation
    else:
        current_state = "chat"

    # =============================================================
    # STEP 6 — PROCESSING & RETRIEVAL FLAGS
    # =============================================================
    if current_state == "blocked":
        requires_retrieval = False
    elif is_affirmation_response:
        requires_retrieval = False
    elif current_state == "clinical":
        requires_retrieval = True
    else:
        requires_retrieval = False

    logger.info(
        "🩺 Dialogue State: [%s] | Scope: %s (%s) | Retrieval: %s | Subject: %s",
        current_state,
        scope,
        scope_source,
        requires_retrieval,
        clinical_subject,
    )

    return {
        "is_safe": current_state != "blocked",
        "dialog_state": current_state,
        "scope": scope,
        "scope_source": scope_source,
        "decision": scope_debug.get("decision", "HIGH_CONFIDENCE"),
        "requires_retrieval": requires_retrieval,
        "clinical_subject": clinical_subject,
        "followup_pending": followup_pending_out,
        "agent_response": agent_response,
        "user_query": user_query,
        "image_path": None,
        "process_image": False,
        "routing_trace": trace + [
            f"dialogue_mgr: "
            f"scope={scope} (source={scope_source}"
            + (
                f", score={scope_debug.get('best_score')}, "
                f"margin={scope_debug.get('margin')}"
                if scope_debug else ""
            )
            + ") | "
            f"state={current_state} | "
            f"retrieval={requires_retrieval} | "
            f"subject={clinical_subject}"
        ],
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
    start_time = time.perf_counter()
    
    # Strictly check process_image authorization flag from Dialogue Manager
    if not state.get("process_image", False):
        logger.info("⏩ Skipping image preprocessing (Turn 2+ or text-only turn).")
        return {"processed_image_payload": None, "image_proc_latency_ms": 0.0}

    image_path = state.get("image_path")
    if not image_path or not Path(image_path).exists():
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "is_safe": False,
            "dialog_state": "blocked",
            "processed_image_payload": None,
            "agent_response": "I couldn't locate the attached image file. Please re-upload a valid medical image.",
            "image_proc_latency_ms": elapsed_ms
        }
        
    try:
        proc_result = image_processor.process(image_path)
        image_payload = proc_result.image_base64 if proc_result else None

        if not image_payload or not str(image_payload).strip():
            raise ValueError("Image processor returned an empty payload.")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info("⏱️ Image Preprocessing Complete | Latency: %.2f ms", elapsed_ms)

        return {
            "image_path": image_path,
            "processed_image_payload": image_payload,
            "image_proc_latency_ms": elapsed_ms
        }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.error("❌ Image preprocessing failed after %.2f ms: %s", elapsed_ms, e)
        return {
            "is_safe": False,
            "dialog_state": "blocked",
            "processed_image_payload": None,
            "agent_response": "I couldn't process the attached image. Please re-upload a supported image file.",
            "image_proc_latency_ms": elapsed_ms
        }

# In src/agent/nodes.py

def run_visual_subject_extraction(state: AgentState) -> Dict[str, Any]:
    """Extract a tentative visual subject before clinical retrieval (Stage 1 Pass).

    This node executes a lightweight, fast visual inspection pass (num_predict=128)
    using the Qwen Vision model to identify key visual features from uploaded images.
    It operates under strict safety guardrails:
    1. Describes visible features using 1-3 words of observational language.
    2. Explicitly avoids confirming diagnostic labels or inferring diseases.
    3. Handles non-medical or ambiguous images safely by returning `None` for the 
       clinical subject, allowing downstream retrieval to fall back cleanly to 
       the user's text query.

    Args:
        state (AgentState): The active LangGraph workflow state containing:
            - requires_retrieval (bool): Authorization flag for evidence grounding.
            - dialog_state (str): Current FSM state (must be "multimodal_triage").
            - processed_image_payload (str): Base64-encoded image string.

    Returns:
        Dict[str, Any]: State update dictionary containing:
            - tentative_subject (Optional[str]): Cleaned observational term or None.
            - subject_confidence (str): "high", "medium", or "low".
            - clinical_subject (Optional[str]): Preserved subject key for vector search.
            - visual_extraction_latency_ms (float): Execution duration in milliseconds.
    """
    logger.info("=== [Node: Visual Subject Extraction (Stage 1)] ===")

    # ------------------------------------------------------------------
    # 1. Guard Clause: Skip visual extraction if conditions are not met
    # ------------------------------------------------------------------
    if (
        not state.get("requires_retrieval", False)
        or state.get("dialog_state") != "multimodal_triage"
        or not state.get("processed_image_payload")
    ):
        logger.info("⏩ Skipping visual subject extraction (non-guidance turn or missing payload).")
        return {}

    # Extract and clean the Base64 image payload (strips 'data:image/...;base64,' prefix if present)
    image_payload = state["processed_image_payload"]
    clean_base64 = (
        str(image_payload).split("base64,", 1)[-1]
        if "base64," in str(image_payload)
        else str(image_payload)
    )

    # ------------------------------------------------------------------
    # 2. System & User Prompts (Observational Language & Scope Limits)
    # ------------------------------------------------------------------
    prompt = """
Analyze the image cautiously to identify a tentative visible finding
relevant to the user's clinical question.

Rules:
1. Do not provide a confirmed diagnosis.
2. If the image is clearly non-medical, return "non_medical".
3. If a possible medical finding cannot be identified reliably, return "uncertain".
4. If a clear clinical or dermatological finding is visible, summarize it in 1-3 words (e.g., "acne vulgaris", "skin rash", "erythema" or "visible swelling").
5. Do not infer a disease solely from an image.
6. Return only a valid JSON object in this format:
{
  "tentative_subject": "short finding, uncertain, or non_medical",
  "confidence": "high, medium, or low"
}
"""

    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You cautiously describe visible image findings. "
                    "Never claim a confirmed diagnosis. "
                    "Return valid JSON only."
                ),
            },
            {
                "role": "user",
                "content": prompt,
                "images": [clean_base64],
            },
        ],
        "think": False,
        "options": {
            "temperature": 0.1,    # Low temperature for deterministic JSON output
            "num_predict": 128,     # Low token budget for fast Stage 1 pass (~1.2s)
        },
        "stream": False,
    }

    start = time.perf_counter()

    try:
        # ------------------------------------------------------------------
        # 3. Ollama API Chat Post Request with Read Timeout
        # ------------------------------------------------------------------
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=(10, 60),  # (connect_timeout=10s, read_timeout=60s)
        )
        response.raise_for_status()

        raw = response.json().get("message", {}).get("content", "").strip()
        if not raw:
            raise ValueError("Empty response from visual model.")

        # ------------------------------------------------------------------
        # 4. Dual-Stage Resilient JSON Parsing
        # ------------------------------------------------------------------
        # Pass 1: Clean out Markdown code fences (e.g., ```json ... ```)
        cleaned = re.sub(
            r"\A```(?:json)?\s*|\s*```\Z",
            "",
            raw,
            flags=re.IGNORECASE,
        ).strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # Pass 2: Fallback regex to extract JSON object embedded in conversational text
            match = re.search(r"\{[\s\S]*\}", cleaned)
            if not match:
                raise ValueError("No valid JSON object structure found in model output.")
            parsed = json.loads(match.group(0))

        if not isinstance(parsed, dict):
            raise ValueError("Parsed model response is not a valid JSON dictionary.")

        # ------------------------------------------------------------------
        # 5. Extract & Normalize Subject & Confidence Values
        # ------------------------------------------------------------------
        subject = parsed.get("tentative_subject")
        confidence = str(parsed.get("confidence", "low")).lower().strip()

        if not isinstance(subject, str) or not subject.strip():
            raise ValueError("Field 'tentative_subject' is missing or empty in JSON.")

        if confidence not in {"high", "medium", "low"}:
            confidence = "low"

        subject = subject.strip().lower()

        # ------------------------------------------------------------------
        # 6. Scope Boundary Normalization (Non-Medical / Uncertain Images)
        # ------------------------------------------------------------------
        if subject in {
            "uncertain",
            "non_medical",
            "non-medical",
            "none",
        }:
            logger.info("ℹ️ Stage 1 Extractor identified a non-medical or uncertain image.")
            # Normalizing to None allows vector_search() to fall back to the user query
            subject = None
            confidence = "low"

        latency_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "✅ Stage 1 Visual Extraction Complete | subject='%s' | "
            "confidence='%s' | latency=%.1f ms",
            subject,
            confidence,
            latency_ms,
        )

        return {
            "tentative_subject": subject,
            "subject_confidence": confidence,
            "clinical_subject": subject,
            "visual_extraction_latency_ms": latency_ms,
        }

    except (
        requests.RequestException,
        ValueError,
        TypeError,
        AttributeError,
    ) as err:
        latency_ms = (time.perf_counter() - start) * 1000

        logger.warning("⚠️ Visual extraction failed safely: %s", err)

        # Defensive Fallback: Never invent or hallucinate a subject on error
        return {
            "tentative_subject": None,
            "subject_confidence": "low",
            "clinical_subject": None,
            "visual_extraction_latency_ms": latency_ms,
        }
        
def run_vector_search(state: AgentState) -> Dict[str, Any]:
    """
    Node 4: Contextual Vector Search.
    Executes ONLY when requires_retrieval == True and resolves query using clinical_subject.
    Applies query normalization to remove conversational structure while retaining clinical terms.
    """
    logger.info("=== [Node: Local Vector Retrieval] ===")
    start_time = time.perf_counter()
    
    if not state.get("requires_retrieval", False):
        logger.info("ℹ️ requires_retrieval is False. Bypassing vector retrieval.")
        return {
            "retrieved_context": [],
            "context_sources": [],
            "top_retrieval_distance": 0.0,
            "retrieval_query": None,
            "vector_search_latency_ms": 0.0
        }

    user_query = state.get("user_query", "").strip()
    clinical_subject = (
        state.get("clinical_subject")
        or state.get("tentative_subject")
    )

    query_lower = user_query.lower()

    # Intent-expanded query construction
    if clinical_subject:
        if any(k in query_lower for k in ("treatment", "treat", "cure", "management", "relief")):
            raw_query = f"{clinical_subject} treatment and management"

        elif any(k in query_lower for k in ("cause", "causes", "why", "trigger", "triggers")):
            raw_query = f"{clinical_subject} causes and triggers"

        elif any(k in query_lower for k in ("symptom", "symptoms", "signs", "warning signs")):
            raw_query = f"{clinical_subject} symptoms and warning signs"

        elif any(k in query_lower for k in ("prevent", "prevention", "avoid")):
            raw_query = f"{clinical_subject} prevention"

        else:
            raw_query = f"{clinical_subject} {user_query}"
    else:
        # Extraction failed: retain original user intent
        raw_query = user_query

    search_query = normalize_query(raw_query)
    logger.info("Targeted RAG query: '%s'", search_query)
    
    search_results = []
    top_distance = 1.0
    
    try:
        logger.info("🔍 Executing Normalized Vector Search | Raw: '%s' -> Clean: '%s'", raw_query, search_query)
        
        # Execute hybrid search with strict distance score ceiling (d <= 0.55)
        search_results = vector_store.hybrid_search(query=search_query, k=3, score_threshold=0.55) or []
        
        if search_results:
            # Extract top distance score for logging/metrics
            top_distance = float(search_results[0].metadata.get("score", 0.35))
            logger.info("📊 Top candidate match distance score: %.4f", top_distance)
        else:
            logger.info("ℹ️ Zero candidates cleared the d <= 0.55 score threshold.")

    except Exception as e:
        logger.exception("❌ Local vector search execution failed: %s", e)

    retrieved_chunks = [doc.page_content for doc in search_results]
    sources = list(dict.fromkeys(doc.metadata.get("source", "verified_doc") for doc in search_results))

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info("⏱️ Vector Retrieval Complete | Latency: %.2f ms | Retained Chunks: %d", elapsed_ms, len(retrieved_chunks))
    
    return {
        "retrieved_context": retrieved_chunks,
        "context_sources": sources,
        "retrieval_query": search_query,
        "top_retrieval_distance": top_distance,
        "vector_search_latency_ms": elapsed_ms
    }


def run_web_search_tool(state: AgentState) -> Dict[str, Any]:
    """
    Node 5: Trusted Web Search Fallback Engine.

    Web retrieval is strictly intent-driven. Executes trusted web retrieval with
    clinical intent normalization, multi-source round-robin interleaving, and 
    validated vector store caching when local retrieval yields zero candidates.
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

    # =============================================================
    # STEP 2 — CONTEXT-AWARE QUERY CONSTRUCTION & INTENT MAPPING
    # =============================================================
    clean_query = normalize_query(user_query)

    if clinical_subject:
        # Strip conversational intent fillers to build focused keyword search queries
        query_lower = user_query.lower()

        if any(k in query_lower for k in [
            "treat", "treatment", "cure", "management", "relief"
        ]):
            search_query = f"{clinical_subject} treatment"

        elif any(k in query_lower for k in [
            "symptom", "symptoms", "sign", "signs", "cause", "causes", "trigger", "triggers"
        ]):
            search_query = f"{clinical_subject} symptoms causes"

        elif any(k in query_lower for k in [
            "prevent", "prevention", "preventive"
        ]):
            search_query = f"{clinical_subject} prevention"

        else:
            search_query = f"{clinical_subject} {clean_query}".strip()
    else:
        search_query = clean_query

    logger.info("🔍 Context-resolved web search query: '%s'", search_query)
    
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

    # =============================================================
    # STEP 5 — SCRAPE, PARSE & CACHE
    # =============================================================
    # Accumulate chunks by source name to preserve source diversity
    source_chunks_map: Dict[str, List[str]] = {}
    context_sources = []

    for url in target_urls:
        logger.info("🌐 Processing trusted URL: %s", url)

        try:
            parsed = urlparse(url)
            parsed_domain = parsed.netloc.replace("www.", "")

            matched_domain = next(
                (d for d in TRUSTED_WEB_SOURCES if d in parsed_domain),
                "Trusted Source"
            )

            # Extract source information early before try/cache blocks
            source_info = TRUSTED_WEB_SOURCES.get(
                matched_domain,
                {"name": "Trusted Source", "score": 7}
            )

            source_name = source_info["name"]

            # -----------------------------------------------------
            # Cache lookup pass
            # -----------------------------------------------------
            try:
                existing_cache = vector_store.vectorstore.get(where={"url": url})

                if existing_cache and existing_cache.get("ids"):
                    cached_docs = existing_cache.get("documents", [])

                    # Quality filter: validate cached chunks before reusing
                    valid_cached = [
                        doc
                        for doc in cached_docs
                        if isinstance(doc, str) and len(doc.strip()) > 120
                    ]

                    if valid_cached:
                        logger.info(
                            "💾 Cache hit. Loading %d validated chunks for: %s",
                            len(valid_cached),
                            url
                        )

                        source_chunks_map.setdefault(source_name, []).extend(valid_cached)

                        if source_name not in context_sources:
                            context_sources.append(source_name)

                        continue

            except Exception as cache_err:
                logger.warning("Cache lookup pass bypassed safely: %s", cache_err)

            # -----------------------------------------------------
            # Web scraping pass
            # -----------------------------------------------------
            raw_chunks = (
                web_scraper.scrape_to_structured_markdown(
                    url,
                    query_condition=clinical_subject or search_query
                )
                or []
            )

            raw_count = len(raw_chunks)
            logger.info("📄 Scraper returned %d raw chunks from %s", raw_count, url)

            # Quality Filter: Purge boilerplate text chunks under 120 characters
            web_chunks = [
                chunk
                for chunk in raw_chunks
                if len(chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)) > 120
            ]

            logger.info("🧹 Quality filter retained %d/%d chunks from %s", len(web_chunks), raw_count, url)

            if not web_chunks:
                logger.warning("⚠️ No usable chunks retained after quality filtering for %s", url)
                continue

            # -----------------------------------------------------
            # Metadata creation & vector DB persistence
            # -----------------------------------------------------
            path_parts = [p for p in parsed.path.split("/") if p]

            condition = (
                "Diet & Nutrition"
                if search_domain == "nutrition"
                else "General Health"
            )

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
                chunk_text = (
                    chunk["text"]
                    if isinstance(chunk, dict)
                    else str(chunk)
                )

                existing_meta = (
                    chunk.get("metadata", {})
                    if isinstance(chunk, dict)
                    else {}
                )

                meta = build_metadata(
                    source=source_name,
                    url=url,
                    condition=existing_meta.get("condition", condition),
                    section=existing_meta.get("section_header", "Overview"),
                    trust_score=source_info["score"],  # Dynamic score lookup (10, 9, 8, or default 7)
                    risk=active_graph_risk,
                    category=(
                        "Diet & Nutrition"
                        if search_domain == "nutrition"
                        else "General Medicine"
                    )
                )

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

                source_chunks_map.setdefault(source_name, []).extend(processed_texts)

                if source_name not in context_sources:
                    context_sources.append(source_name)

        except Exception as page_ex:
            logger.exception("❌ Isolated failure processing web page %s | Reason: %s", url, page_ex)
            continue

    # =============================================================
    # STEP 6 — DIVERSE ROUND-ROBIN INTERLEAVING
    # =============================================================
    MAX_WEB_CHUNKS = 5
    retrieved_context = []
    selected_sources = []

    # Interleave 1 chunk per institutional source until capped or exhausted
    while (
        len(retrieved_context) < MAX_WEB_CHUNKS
        and any(source_chunks_map.values())
    ):
        for source_name in list(source_chunks_map.keys()):
            if not source_chunks_map[source_name]:
                continue

            chunk = source_chunks_map[source_name].pop(0)
            retrieved_context.append(chunk)
            selected_sources.append(source_name)

            if len(retrieved_context) >= MAX_WEB_CHUNKS:
                break

    logger.info(
        "📚 Final web context: %d chunks from sources: %s",
        len(retrieved_context),
        selected_sources
    )
    logger.info(
        "⏱️ Diverse Web Scraper Complete. Latency: %.2f ms | Retained Chunks: %d",
        (time.time() - start_time) * 1000,
        len(retrieved_context)
    )

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
    
    # Preserve pre-generated refusal/crisis text from Guardrails
    existing_response = state.get("agent_response")
    if existing_response:
        return {
                "agent_response": existing_response,
                "context_sources": ["System Safety Guardrail"]
            }
        
        
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
    Node 7: Runs MediQwen local LLM generation via Ollama /api/chat.

    Dynamically selects focused behavioral skills based on the
    authoritative dialogue state and retrieval state.
    """
    logger.info("=== [Node: MediQwen Response Generation] ===")

    # -------------------------------------------------------------
    # 1. Extract & Normalize State Parameters
    # -------------------------------------------------------------
    user_query = state.get("user_query", "").strip()

    retrieved_context = state.get("retrieved_context", []) or []

    dialog_state = str(
        state.get("dialog_state", "chat")
    ).lower().strip()

    requires_retrieval = bool(
        state.get("requires_retrieval", False)
    )

    risk_tier = str(
        state.get("risk_level", "LOW")
    ).upper().strip()

    # Safely create a shallow copy to prevent dictionary mutation across graph nodes
    risk_metadata = dict(state.get("risk_metadata", {}) or {}).copy()

    image_payload = state.get("processed_image_payload")

    clinical_subject = state.get("clinical_subject")

    subject_str = clinical_subject if clinical_subject else "Unspecified"
    
    # Short-Circuit for Authoritative Pre-Populated Responses
    # -------------------------------------------------------------
    prepopulated_response = state.get("agent_response")
    is_affirmation = state.get("is_affirmation_response", False)

    if prepopulated_response and (dialog_state == "blocked" or is_affirmation):
        logger.info(
            "🛑 Preserving pre-populated authoritative response; skipping Ollama generation."
        )
        return {
            "agent_response": prepopulated_response,
            "clinical_subject": clinical_subject,
            "generation_success": True,
            "generation_error": None,
            "generation_latency_ms": 0.0,
        }

    # -------------------------------------------------------------
    # 2. Defensive Evidence Invariant
    # -------------------------------------------------------------
    if requires_retrieval and not retrieved_context:
        logger.error(
            "❌ No approved clinical context | risk=%s",
            risk_tier,
        )
        risk_metadata["is_medical"] = True
        risk_metadata["risk_level"] = risk_tier

        if risk_tier == "EMERGENCY":
            fallback_text = (
                "EMERGENCY: Your symptoms may indicate a life-threatening "
                "condition. Call your local emergency number (such as 911, 112, or 999) "
                "now and follow the dispatcher's instructions. Do not drive yourself if "
                "you may be experiencing a medical emergency.\n\n"
                "I could not retrieve verified clinical information for "
                "this response. Do not delay emergency care while seeking "
                "additional information."
            )
        else:
            fallback_text = (
                "I don't have sufficient approved clinical context "
                "to provide a reliable answer for this specific question."
            )

        return {
            "agent_response": guardrails.apply_output_guardrails(
                fallback_text,
                risk_metadata=risk_metadata,
            ),
            "clinical_subject": clinical_subject,
            "generation_success": False,
            "generation_error": "Missing approved clinical context",
            "generation_latency_ms": 0.0,
        }

    # -------------------------------------------------------------
    # 3. Base64 Image Preparation
    # -------------------------------------------------------------
    clean_base64 = None
    if image_payload:
        raw_payload = str(image_payload)
        clean_base64 = (
            raw_payload.split("base64,", 1)[-1]
            if "base64," in raw_payload
            else raw_payload
        )

    # -------------------------------------------------------------
    # 4. Base System Identity
    # -------------------------------------------------------------
    base_system_prompt = """
You are MediQwen, an evidence-grounded clinical AI assistant.

Your role is to assist with health-related questions and clinical information.

System safety, risk classification, routing decisions, and guardrails are authoritative.
""".strip()

    # -------------------------------------------------------------
    # 5. Dynamic Skill Selection
    # Hierarchy: CHAT -> MULTIMODAL_TRIAGE -> EVIDENCE_SYNTHESIS -> CLINICAL_TRIAGE
    # -------------------------------------------------------------
    active_skill_prompt = ""

    if dialog_state == "chat":
        active_skill_prompt = load_skill("casual_chat")
    elif dialog_state == "multimodal_triage":
        active_skill_prompt = load_skill("multimodal_triage")
    elif requires_retrieval:
        active_skill_prompt = load_skill("evidence_synthesis")
    elif dialog_state == "clinical":
        active_skill_prompt = load_skill("clinical_triage")

    # -------------------------------------------------------------
    # 6. Format Retrieved Evidence
    # -------------------------------------------------------------
    context_str = ""
    if retrieved_context:
        formatted_docs = "\n\n".join(
            (
                doc.page_content
                if hasattr(doc, "page_content")
                else str(doc)
            )
            for doc in retrieved_context
        )

        context_str = (
            "\n\n=== APPROVED CLINICAL CONTEXT ===\n"
            f"{formatted_docs}"
        )

    # -------------------------------------------------------------
    # 7. Assemble System Instruction (Strict Trust Boundary)
    # -------------------------------------------------------------
    system_instruction = base_system_prompt

    if active_skill_prompt:
        system_instruction += (
            "\n\n=== ACTIVE OPERATIONAL SKILL ===\n"
            f"{active_skill_prompt}"
        )

    # Only attach active clinical state metadata when not in casual chat
    if dialog_state != "chat":
        system_instruction += (
            f"\n\nACTIVE DIALOGUE STATE: {dialog_state}"
            f"\nACTIVE TRIAGE RISK: {risk_tier}"
            f"\nCLINICAL SUBJECT: {subject_str}"
            f"{context_str}"
        )

    # -------------------------------------------------------------
    # 8. Build Ollama Chat Payload
    # Allow larger output buffers for visual triage and emergency escalations
    max_tokens = 768 if (dialog_state == "multimodal_triage" or risk_tier == "EMERGENCY") else 512

    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_query},
        ],
        "think": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 4096,
            "num_predict": max_tokens, # 👈 Prevents truncation on detailed visual/emergency turns
        },
        "stream": False,
    }

    # -------------------------------------------------------------
    # 9. Attach Image Payload for Multimodal Triage
    # -------------------------------------------------------------
    if clean_base64 and dialog_state == "multimodal_triage":
        payload["messages"][1]["images"] = [clean_base64]

    # -------------------------------------------------------------
    # 10. Execute Ollama /api/chat Request
    # -------------------------------------------------------------
    generation_start = time.perf_counter()

    try:
        logger.info(
            "🤖 Starting Ollama chat | "
            "model=%s | "
            "system_chars=%d | "
            "user_chars=%d | "
            "context_chunks=%d | "
            "think=False",
            payload["model"],
            len(system_instruction),
            len(user_query),
            len(retrieved_context),
        )

        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=(10, 600),
        )

        response.raise_for_status()
        response_data = response.json()

        message = response_data.get("message", {})
        generated_text = message.get("content", "").strip()

        generation_latency_ms = (
            time.perf_counter() - generation_start
        ) * 1000.0

        logger.info(
            "⏱️ Ollama generation completed: %.2f ms",
            generation_latency_ms,
        )

        logger.info(
            "🔍 Ollama response | "
            "done=%s | "
            "reason=%s | "
            "content_chars=%d | "
            "eval_count=%s",
            response_data.get("done"),
            response_data.get("done_reason"),
            len(generated_text),
            response_data.get("eval_count"),
        )

        # ---------------------------------------------------------
        # Dialogue-Aware Empty Response Recovery
        # ---------------------------------------------------------
        if not generated_text:
            logger.error("❌ Ollama returned empty content.")

            if dialog_state == "chat":
                generated_text = (
                    "Hello! I'm MediQwen, a clinical AI assistant. "
                    "How can I help with your health questions today?"
                )
            elif risk_tier == "EMERGENCY":
                generated_text = (
                    "Your symptoms may require urgent medical attention. "
                    "Please seek emergency medical care immediately."
                )
            else:
                generated_text = (
                    "I was unable to generate a reliable response. "
                    "Please try again or rephrase your question."
                )

        # ---------------------------------------------------------
        # Multimodal Subject Extraction
        # ---------------------------------------------------------
        extracted_subject = clinical_subject

        if dialog_state == "multimodal_triage" and not extracted_subject:
            gen_lower = generated_text.lower()
            for pattern in MEDICAL_SUBJECT_PATTERNS:
                match = re.search(pattern, gen_lower)
                if match:
                    extracted_subject = match.group(1)
                    logger.info(
                        "🏷️ Extracted clinical subject: %s",
                        extracted_subject,
                    )
                    break

        # ---------------------------------------------------------
        # Output Guardrail Application
        # ---------------------------------------------------------
        risk_metadata["is_medical"] = (
            dialog_state in {"clinical", "multimodal_triage"}
        )
        risk_metadata["risk_level"] = risk_tier

        final_safe_text = guardrails.apply_output_guardrails(
            generated_text,
            risk_metadata=risk_metadata,
        )

        return {
            "agent_response": final_safe_text,
            "clinical_subject": extracted_subject,
            "generation_success": True,
            "generation_error": None,
            "generation_latency_ms": generation_latency_ms,
        }

    except Exception as err:
        generation_latency_ms = (
            time.perf_counter() - generation_start
        ) * 1000.0

        logger.exception(
            "❌ Ollama generation failed after %.2f ms | error=%s",
            generation_latency_ms,
            str(err),
        )

        # ---------------------------------------------------------
        # Dialogue-Aware Exception Fallback
        # ---------------------------------------------------------
        if dialog_state == "chat":
            fallback_text = (
                "Sorry, I couldn't generate a response right now. "
                "Please try again."
            )
        elif risk_tier == "EMERGENCY":
            fallback_text = (
                "Your symptoms may require urgent medical attention. "
                "Please seek emergency medical care immediately."
            )
        else:
            fallback_text = (
                "I was unable to generate a reliable response. "
                "Please try again."
            )

        risk_metadata["is_medical"] = (
            dialog_state in {"clinical", "multimodal_triage"}
        )
        risk_metadata["risk_level"] = risk_tier

        final_fallback = guardrails.apply_output_guardrails(
            fallback_text,
            risk_metadata=risk_metadata,
        )

        return {
            "agent_response": final_fallback,
            "clinical_subject": clinical_subject,
            "generation_success": False,
            "generation_error": str(err),
            "generation_latency_ms": generation_latency_ms,
        }