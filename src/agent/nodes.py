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
from src.tools.helpers import normalize_query, detect_search_domain, build_metadata, is_generic_vision_query
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
    Node 1: Hardened Dialogue Manager State Machine (FSM).
    Manages state persistence, interprets linguistic categories, and handles
    graceful context closure loops without sudden drops.
    """
    logger.info("=== [Node: Dialogue Manager FSM] ===")
    
    user_query = state.get("user_query", "").strip()
    image_path = state.get("image_path")
    query_lower = user_query.lower()
    
    previous_state = state.get("dialog_state", "chat").lower().strip()
    trace = state.get("routing_trace", []) or []

    # 1. Word-safe primary intent extraction loop
    query_tokens = set(TOKEN_REGEX.findall(query_lower))
    has_medical_concepts = any(
        re.search(rf"\b{re.escape(keyword)}\b", query_lower)
        for keyword in guardrails.MEDICAL_HEALTH_KEYWORDS
    )
    is_nutrition_query = bool(query_tokens & NUTRITION_KEYWORDS)

    # 2. Safety Infraction Short-Circuit
    if not is_nutrition_query:
        is_safe, output_text = guardrails.validate_input(user_query)
        if not is_safe:
            return {
                "is_safe": False,
                "dialog_state": "blocked",
                "agent_response": output_text,
                "routing_trace": trace + ["dialogue_mgr -> blocked_refusal"]
            }

    # 3. Structural Linguistic Evaluation Pass
    is_structural_followup = any(re.search(p, query_lower) for p in guardrails.STRUCTURAL_FOLLOWUPS)
    is_explicit_topic_switch = any(re.search(p, query_lower) for p in guardrails.EXPLICIT_TOPIC_SWITCHES)
    is_closure_phrase = any(re.search(p, query_lower) for p in guardrails.CLOSURE_EXPRESSIONS)

    current_state = previous_state

    # CASE 1: Currently inside an ongoing Clinical/Nutrition Session
    if previous_state in ["clinical", "emergency"]:
        if is_explicit_topic_switch:
            current_state = "chat"
            logger.info("📉 FSM Context Shift: Explicit topic switch requested. Transitioning to CHAT.")
        elif is_closure_phrase:
            current_state = "closing"
            logger.info("🏁 FSM Context Transition: Gratitude/Closure spotted. Transitioning to CLOSING.")
        else:
            current_state = "clinical"
            logger.info("🩺 FSM Sticky State: Retaining clinical/nutrition execution loop.")

    # CASE 2: Currently inside a Graceful Closing Gate
    elif previous_state == "closing":
        if has_medical_concepts or is_nutrition_query or is_structural_followup:
            current_state = "clinical"
            logger.info("🩺 FSM Re-escalation: New clinical context provided. Returning to CLINICAL.")
        elif is_closure_phrase or is_explicit_topic_switch or len(query_lower.split()) <= 2:
            current_state = "chat"
            logger.info("📉 FSM Exit: Wrap-up complete. Returning to baseline casual CHAT.")
        else:
            current_state = "clinical"
            logger.info("🩺 FSM Protective Assumption: Ambiguous follow-up. Retaining CLINICAL focus.")

    # CASE 3: Baseline Casual Chat Space
    else:
        if has_medical_concepts or is_nutrition_query:
            current_state = "clinical"
            logger.info("🩺 FSM Promotion: Direct medical/nutrition indicator detected. Transitioning to CLINICAL.")
        else:
            current_state = "chat"

    # --- 📸 IMAGE AUTHORIZATION LOGIC ---
    # Auto-promote to clinical if an image is provided alongside casual chat
    if image_path and current_state == "chat":
        current_state = "clinical"
        logger.info("📸 Image payload detected in casual state. Promoting request to CLINICAL / VISION pipeline.")

    # Determine whether downstream vision nodes should run
    IMAGE_ENABLED_STATES = {"clinical", "emergency"}
    process_image = (
        image_path is not None 
        and current_state in IMAGE_ENABLED_STATES
    )

    return {
        "is_safe": True,
        "dialog_state": current_state,
        "agent_response": "",
        "user_query": user_query,
        "image_path": image_path if process_image else None,
        "process_image": process_image,
        "routing_trace": trace + [
            f"dialogue_mgr: current={current_state} (prev={previous_state}) | process_image={process_image}"
        ]
    }
    
def run_risk_classification(state: AgentState) -> Dict[str, Any]:
    """Node 2: Runs triage on queries using rule-based criteria."""
    logger.info("=== [Node: Risk Classification] ===")
    user_query = state.get("user_query", "")
    classification = risk_classifier.classify_query(user_query)
    
    return {
        "risk_level": classification.get("risk_level", "low"),
        "risk_metadata": classification
    }

def run_multimodal_processing(state: AgentState) -> Dict[str, Any]:
    """Node 3: Preprocesses image assets into clean Base64 frames for Gemma 4 E2B native vision."""
    logger.info("=== [Node: Multimodal Processing] ===")
    
    # 1. Respect FSM image processing authorization flag
    if not state.get("process_image", False):
        logger.info("⏩ Skipping multimodal processing (FSM state bypass or no image attached).")
        return {"processed_image_payload": None}

    image_path = state.get("image_path")
    
    if not image_path or not Path(image_path).exists():
        logger.warning(f"⚠️ Target file path missing or does not exist on disk: {image_path}")
        return {"image_path": None, "processed_image_payload": None}
        
    try:
        logger.info(f"📸 Image asset authorized for vision processing: {image_path}")
        
        # Route to CLAHE skin enhancement or clean scaling depending on file naming
        modality = "dermatology" if "derma" in image_path.lower() else "general"
        proc_result = image_processor.process(image_path, modality=modality)
        
        # Track that an image source was ingested
        context_sources = list(state.get("context_sources", []))
        context_sources.append(f"Visual Payload ({proc_result.modality})")
        
        return {
            "image_path": image_path,
            "processed_image_payload": proc_result.image_base64,
            "context_sources": context_sources
        }
    except Exception as e:
        logger.error(f"❌ Image Preprocessing Node failure: {e}")
        return {"processed_image_payload": None}
    

def run_vector_search(state: AgentState) -> Dict[str, Any]:
    """Node 4: Queries ChromaDB local collection with latency logging."""
    logger.info("=== [Node: Local Vector Retrieval] ===")
    start_time = time.time()  
    user_query = state.get("user_query", "")
    
    has_image = bool(state.get("processed_image_payload")) or bool(state.get("image_path"))

    # 🛑 Pure Vision Bypass: If generic query + image, skip vector retrieval entirely
    if has_image and is_generic_vision_query(user_query, GENERIC_IMAGE_PHRASES):
        logger.info("👁️ Pure Vision Mode: Generic query with attached image detected. Skipping local vector retrieval.")
        return {
            "retrieved_context": [],
            "top_retrieval_distance": 0.0
        }
        
    search_results = []
    top_retrieval_distance = 1.0
    
    try:
        search_results = vector_store.hybrid_search(query=user_query, k=3, score_threshold=0.6)
        if search_results:
            top_retrieval_distance = float(search_results[0].metadata.get("score", 0.35))
    except Exception as e:
        logger.error(f"❌ Vector search failed: {e}")

    retrieved_chunks = []
    context_sources = []

    for doc in search_results:
        retrieved_chunks.append(doc.page_content)
        source_name = doc.metadata.get("source", "verified_document")
        if source_name not in context_sources:
            context_sources.append(source_name)

    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(f"⏱️ Retrieval Complete. Latency: {elapsed_ms:.2f} ms | Located Chunks: {len(retrieved_chunks)}")
    
    return {
        "retrieved_context": retrieved_chunks,
        "context_sources": context_sources,
        "risk_level": state.get("risk_level", "low"),
        "top_retrieval_distance": top_retrieval_distance
    }

def run_conversational_skill_node(state: AgentState) -> Dict[str, Any]:
    """Skill Node A: Handles casual dialogue frames safely utilizing model isolation layers."""
    logger.info("=== [Skill Node: Conversational Engine] ===")
    user_query = state.get("user_query", "")
    
    system_prompt = (
        "You are MediQwen's general conversation skill mode. Your purpose is to greet users warmly, "
        "answer friendly introductions, and acknowledge pleasantries. Always remain polite and professional.\n"
        "CRITICAL BOUNDARIES:\n"
        "1. Do NOT answer complex general knowledge questions unrelated to health or your identity.\n"
        "2. Do NOT write software code, scripts, or explain malware/exploits.\n"
        "3. If asked about health issues, politely ask them to state it as a direct medical inquiry."
    )
    
    full_prompt = f"System: {system_prompt}\n\nUser: {user_query}\n\nResponse:"
    ollama_url = "http://localhost:11434/api/generate"
    try:
        response = requests.post(
            ollama_url,
            json={
                "model": "mediqwen:latest", 
                "prompt": full_prompt, 
                "stream": False,
                "options": {"temperature": 0.5}
            },
            timeout=120
        )
        
        if response.status_code == 200:
            raw_response = response.json().get("response", "").strip()
        else:
            logger.error(f"❌ Ollama conversational error status: {response.status_code}")
            raw_response = "Hello! I am here to chat, but my main specialty is providing clinical medical guidance."
            
    except Exception as e:
        logger.error(f"❌ Chat generation failed: {e}")
        raw_response = "Hello! Let's keep things brief. How can I help you with your health questions today?"

    disclaimer = "\n\n---\n💬 *Currently executing inside General Chat Skill. For clinical questions, state your medical inquiry.*"
    return {
        "agent_response": raw_response + disclaimer,
        "context_sources": ["Conversational Skill Matrix"]
    }

def run_web_search_tool(state: AgentState) -> Dict[str, Any]:
    """
    Node 5: Upgraded Trusted Web Search Fallback Engine.
    Implements prioritized sorting, predictive cache checks, string tokenization,
    and granular exception handling frames.
    """
    logger.info("=== [Node: Live Web Search Fallback] ===")
    start_time = time.time()
    user_query = state.get("user_query", "")
    
    has_image = bool(state.get("processed_image_payload")) or bool(state.get("image_path"))

    if has_image and (not user_query or is_generic_vision_query(user_query, GENERIC_IMAGE_PHRASES)):
        logger.info("👁️ Pure Vision Bypass Triggered: Generic query with attached image detected.")
        return {
            "retrieved_context": ["Direct visual evaluation requested for uploaded medical asset."],
            "context_sources": ["Direct Visual Payload"]
        }

    # 1. Structured Regex Query Optimization Pass
    clean_query = normalize_query(user_query)
    logger.info(f"Optimized search query text: '{user_query}' → '{clean_query}'")

    # 2. Token-Based Domain Detection
    search_domain = detect_search_domain(clean_query)

    if search_domain == "nutrition":
        logger.info("🥗 Scoping search intent execution to nutrition domains.")
        search_target = f"{clean_query} diet nutrition"
    else:
        logger.info("🩺 Scoping search intent execution to clinical domains.")
        search_target = clean_query

    # Discover target strings using the domain_type parameter
    target_urls = web_scraper.fetch_whitelisted_urls(
        query=clean_query, 
        max_results=4, 
        domain_type=search_domain
    ) or []

    # 3. Dynamic Domain Authority Sorting
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
            "risk_level": state.get("risk_level", "low")
        }

    retrieved_context = []
    context_sources = []
    active_graph_risk = state.get("risk_level", "low")

    # 4. Isolated Failure-Protected Scrape & Cache Execution Loop
    for url in target_urls:
        try:
            # Check vector collection using the index payload parameters to avoid dupe writes
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

            # Execute web scrapers
            web_chunks = web_scraper.scrape_to_structured_markdown(url, query_condition=user_query) or []
            
            # 5. Quality Filter: Purge headers and boilerplate frames under 120 characters
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

            for chunk in web_chunks[:3]:  # Limit to first 3 chunks
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
                
                # Compute persistent SHA-256 Checksum IDs to ensure absolute session persistence
                chunk_id = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                
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
            logger.exception(f"❌ Isolated failure processing target page endpoint: {url} | Reason: {page_ex}")
            continue
    
    # 6. Latency Logging & Final Return Structure
    MAX_WEB_CHUNKS = 3
    if len(retrieved_context) > MAX_WEB_CHUNKS:
        logger.info(f"🛑 Capping retrieved web chunks to {MAX_WEB_CHUNKS} for RAG input. Original count: {len(retrieved_context)}")
        retrieved_context = retrieved_context[:MAX_WEB_CHUNKS]
        
    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(f"⏱️ Web Scraper Complete. Latency: {elapsed_ms:.2f} ms | Located Chunks: {len(retrieved_context)}")

    if search_domain == "nutrition":
        return {
            "retrieved_context": retrieved_context,
            "context_sources": context_sources,
            "risk_level": "",
            "show_risk_badge": False
        }

    return {
        "retrieved_context": retrieved_context,
        "context_sources": context_sources,
        "risk_level": active_graph_risk,
        "show_risk_badge": True
    }

def run_safe_refusal_fallback(state: AgentState) -> Dict[str, Any]:
    """Node 6: Hand-crafted static safety fallback boundary."""
    logger.info("=== [Node: Offline Safe Refusal Fallback] ===")
    user_query = state.get("user_query", "")
    
    refusal_text = (
        f"I currently do not have the verified institutional clinical documentation for '{user_query}' "
        "stored in my offline knowledge base. For your safety, I am restricted from providing diagnostic "
        "or treatment parameters without a verified local reference resource."
    )
    
    return {
        "agent_response": refusal_text,
        "risk_level": "blocked",             
        "context_sources": ["System Safety Exclusion Guard"],
        "retrieved_context": []
    }

def run_qwen_generation(state: AgentState) -> Dict[str, Any]:
    """Node 7: Multi-Modal Context-Budget Aware Response Generation utilizing mediqwen:latest."""
    logger.info("=== [Node: MediQwen Response Generation] ===")
    
    user_query = state.get("user_query", "")
    image_payload = state.get("processed_image_payload")
    risk_tier = state.get("risk_level", "low").upper()
    
    # 1. Build the Multimodal Instruction Header
    image_notice = ""
    if image_payload:
        logger.info("📸 Image payload detected. Prepending Skill 1 Multimodal instructions.")
        image_notice = (
            "### 👁️ MULTIMODAL VISION MODE ACTIVE\n"
            "An image has been attached to this request. "
            "Activate 'Skill 1: Multimodal Evidence Integration'. "
            "Examine the visual characteristics directly from the attached image payload.\n\n"
        )
        
    # 2. Assemble RAG Context Block with strict character limit enforcement
    MAX_CONTEXT_CHARS = 16000
    retrieved_chunks: List[Any] = state.get("retrieved_context", [])
    
    compiled_context_pieces = []
    current_char_accumulation = 0
    
    for i, chunk in enumerate(retrieved_chunks):
        chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)
        source_name = getattr(chunk, "source", None) or chunk.metadata.get("source", "Unknown Asset Reference") if hasattr(chunk, "metadata") else f"Reference Document Segment {i+1}"
        
        formatted_entry = f"Source: {source_name}\nEvidence:\n{chunk_text}"
        entry_length = len(formatted_entry)
        
        if current_char_accumulation + entry_length > MAX_CONTEXT_CHARS:
            logger.info(f"🛑 Context window boundary hit. Capping RAG input stream at element index: {i}")
            break
            
        compiled_context_pieces.append(formatted_entry)
        current_char_accumulation += entry_length + 2

    retrieved_context_str = "\n\n".join(compiled_context_pieces)

    # 3.Assemble Content Prompt
    # -------------------------------------------------------------
    if image_payload:
        # If an image is present, instruct the model to analyze it directly
        user_content_prompt = (
            f"{image_notice}[Attached Medical Image] Please analyze the visual features in the provided image.\n\n"
            f"Patient Query: {user_query}\n\n"
            "Describe what you observe clinically in the photo."
        )
    else:
        user_content_prompt = f"""{image_notice}[Active Runtime Context Block]

    ### AUTHORITATIVE STATE DATA
    Active Triage Risk Category: [{risk_tier}]

    ### RETRIEVED MEDICAL DOCUMENTATION
    {retrieved_context_str if retrieved_context_str else "No target primary data documentation located inside local indexes."}

    ### PATIENT INQUIRY
    Patient Query: "{user_query}"

    ### 🛑 CRITICAL OUTPUT FORMATTING BLUEPRINT
    1. Lead with the clinical answer immediately. Do NOT repeat or echo the patient's query back to them.
    2. Ground your explanation on direct visual observations from the image asset or retrieved context.
    3. If the image is non-clinical (e.g. machinery, vehicles), state politely that it is non-medical.
    """

    # -------------------------------------------------------------
    # 2. Prepare Payload & Process Base64 Data
    # -------------------------------------------------------------
    raw_payload = state.get("processed_image_payload") or image_payload or ""
    clean_base64 = raw_payload.split("base64,")[-1] if "base64," in raw_payload else raw_payload

    payload = {
        "model": "mediqwen:latest",
        "prompt": user_content_prompt,
        "options": {
            "temperature": 0.2,
            "num_ctx": 8192
        },
        "stream": False
    }

    if clean_base64:
        payload["images"] = [clean_base64]

        # Save disk debug copy
        try:
            decoded = Image.open(io.BytesIO(base64.b64decode(clean_base64)))
            decoded.save("debug_sent_to_ollama.png")
            logger.info(f"🐛 [DEBUG] Decoded image saved to disk as 'debug_sent_to_ollama.png'. Size: {decoded.size}")
        except Exception as img_err:
            logger.error(f"❌ Failed to decode base64 debug image: {img_err}")

        logger.info(
            f"🔍 FINAL IMAGE CHECK: length={len(clean_base64)}, "
            f"prefix={clean_base64[:30]}..."
        )
        logger.info(f"🐛 Injected Image Payload Length: {len(payload['images'][0])}")

    # -------------------------------------------------------------
    # 3. Dispatch POST Request to /api/generate
    # -------------------------------------------------------------
    try:
        logger.info("🚀 Dispatching generation frame to local Ollama service via unified payload tracks.")
        ollama_url = "http://localhost:11434"
            
        response = requests.post(
            f"{ollama_url}/api/generate", 
            json=payload, 
            timeout=240
        )
        response.raise_for_status()
        result_json = response.json()
        
        return {
            "agent_response": result_json.get("response", "").strip(),
        }
        
    except Exception as err:
        logger.error(f"❌ Ollama Interface Generation Loop Crash: {err}")
        return {
            "agent_response": "An internal system configuration error occurred. If you are experiencing concerning symptoms, please seek professional medical care or local emergency services immediately."
        }