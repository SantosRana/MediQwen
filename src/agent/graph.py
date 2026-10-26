# src/agent/graph.py
"""
Production Orchestrator Graph for MediQwen.
Enforces non-linear, data-driven execution routing, strict validation gates,
and visual-first risk classification.
"""

import logging
from langgraph.graph import StateGraph, START, END
from src.agent.state import AgentState
from src.agent.nodes import (
    run_dialogue_manager, 
    run_risk_classification, 
    run_multimodal_processing, 
    run_vector_search, 
    run_web_search_tool,
    safe_refusal_node,
    run_qwen_generation,
    run_visual_subject_extraction
)

logger = logging.getLogger("graph_orchestrator")

# Instantiate LangGraph overall workflow builder
workflow = StateGraph(AgentState)

# Register All Concrete Working Processing Nodes
workflow.add_node("dialogue_manager", run_dialogue_manager)
workflow.add_node("multimodal_processor", run_multimodal_processing)
workflow.add_node("classifier", run_risk_classification)
workflow.add_node("retriever", run_vector_search)
workflow.add_node("web_search_tool", run_web_search_tool)
workflow.add_node("safe_refusal_node", safe_refusal_node)
workflow.add_node("generator", run_qwen_generation)
workflow.add_node("visual_subject_extractor", run_visual_subject_extraction)

# =======================================================================
# 🌐 CONDITIONAL ROUTING FUNCTIONS
# =======================================================================

def dialogue_fsm_switchboard(state: AgentState) -> str:
    """
    Routes execution based on the Dialogue Manager FSM state.

    - BLOCKED → Safe refusal
    - CHAT → Generator (Bypasses classification & retrieval)
    - CLINICAL → Risk classifier / Multimodal router
    - MULTIMODAL_TRIAGE → Multimodal processor
    """
    active_state = state.get("dialog_state", "chat").lower().strip()

    if active_state == "blocked":
        logger.info("🛑 FSM Switchboard: [BLOCKED] → Safe Refusal")
        return "blocked_refusal"

    if active_state == "chat":
        logger.info("💬 FSM Switchboard: [CHAT] → Generator")
        return "generator"

    # Route multimodal turns to the visual preprocessor first
    if active_state == "multimodal_triage":
        logger.info("📸 FSM Switchboard: [MULTIMODAL] → Pre-processing Image Before Classification")
        return "multimodal_processor"

    if active_state == "clinical":
        logger.info("🩺 FSM Switchboard: [CLINICAL] → Risk Classifier")
        return "classifier"

    logger.warning(
        f"⚠️ Unknown dialogue state [{active_state.upper()}]. Defaulting to Generator."
    )
    return "generator"

def post_classification_router(state: AgentState) -> str:
    """
    Routes after the risk classifier.

    - Multimodal turn (Image present) -> Always run visual subject extractor first (Stage 1)
    - Text requiring retrieval -> retriever
    - Text casual conversation -> generator
    """
    dialog_state = state.get("dialog_state")
    requires_retrieval = state.get("requires_retrieval", False)
    has_image = bool(state.get("processed_image_payload"))

    if dialog_state == "multimodal_triage" or has_image:
        logger.info("📸 Multimodal turn detected -> Routing to Visual Subject Extractor (Stage 1)")
        return "visual_subject_extractor"

    if requires_retrieval:
        logger.info("🔎 Retrieval required -> Retriever")
        return "retriever"

    logger.info("💬 Retrieval not required -> Generator")
    return "generator"


def knowledge_base_router(state: AgentState) -> str:
    """
    Evaluates intent-driven retrieval status and enforces evidence grounding.
    Allows online emergency requests to attempt trusted web retrieval before falling back to generator.
    """
    trace = state.get("routing_trace", [])
    requires_retrieval = state.get("requires_retrieval", False)
    context = state.get("retrieved_context", [])
    is_emergency = str(state.get("risk_level", "LOW")).upper() == "EMERGENCY"
    is_online = state.get("is_online", False)

    # 1. Deferred RAG Short-Circuit: Skip retrieval if not requested
    if not requires_retrieval:
        logger.info("⏩ Retrieval not requested for this turn. Short-circuiting directly to Generator.")
        state["routing_trace"] = trace + ["retriever -> no_retrieval_needed -> generator"]
        return "generator"

    # 2. Context Found: Proceed to grounded generation
    if context:
        logger.info(
            "✅ RAG context located (%d chunks). Routing to generator.",
            len(context),
        )
        state["routing_trace"] = trace + ["retriever -> context_found -> generator"]
        return "generator"

    # 3. Local retrieval missed: attempt trusted web retrieval when online
    if is_online:
        logger.warning(
            "🌐 Local retrieval miss | emergency=%s | trying web fallback",
            is_emergency,
        )
        state["routing_trace"] = trace + ["retriever -> db_miss_online -> web_search_tool"]
        return "web_search_tool"

    # 4. Offline emergency: no web fallback is possible. Route to emergency generator fallback.
    if is_emergency:
        logger.warning(
            "🚨 Emergency retrieval miss in offline mode. "
            "Routing to emergency-safe generation fallback."
        )
        state["routing_trace"] = trace + ["retriever -> emergency_db_miss_offline -> generator"]
        return "generator"

    # 5. Offline non-emergency miss: enforce safe refusal
    logger.warning("🛡️ Offline retrieval miss. Routing to safe refusal.")
    state["routing_trace"] = trace + ["retriever -> db_miss_offline -> safe_refusal_node"]
    return "safe_refusal_node"


def web_search_validation_router(state: AgentState) -> str:
    """Validate web retrieval and preserve emergency-safe handling."""
    trace = state.get("routing_trace", [])
    context = state.get("retrieved_context", [])
    is_emergency = str(state.get("risk_level", "LOW")).upper() == "EMERGENCY"

    if context:
        state["routing_trace"] = trace + ["web_search_tool -> web_context_found -> generator"]
        return "generator"

    if is_emergency:
        logger.error(
            "🚨 Local and web retrieval failed for an emergency. "
            "Routing to emergency-safe generation fallback."
        )
        state["routing_trace"] = trace + ["web_search_tool -> emergency_web_miss -> generator"]
        return "generator"

    logger.error("❌ Web retrieval returned no verified documents.")
    state["routing_trace"] = trace + ["web_search_tool -> empty_web_miss -> safe_refusal_node"]
    return "safe_refusal_node"

def post_visual_extraction_router(state: AgentState) -> str:
    """Route after visual subject extraction based on image validation status and retrieval requirements."""
    if state.get("image_validation_status") == "non_medical" or not state.get("requires_retrieval", False):
        return "generator"
    
    return "retriever"

# =======================================================================
# 🔗 GRAPH EDGE BINDINGS
# =======================================================================
# 1. Mandatory Graph Entrypoint Edge
workflow.add_edge(START, "dialogue_manager")

# 2. Master FSM Switchboard Edge Routing
workflow.add_conditional_edges(
    "dialogue_manager",
    dialogue_fsm_switchboard,
    {
        "blocked_refusal": "safe_refusal_node",
        "generator": "generator",
        "multimodal_processor": "multimodal_processor",
        "classifier": "classifier",
    }
)

workflow.add_edge("multimodal_processor", "classifier")

# 3. Post-Classification Conditional Routing
workflow.add_conditional_edges(
    "classifier",
    post_classification_router,
    {
        "visual_subject_extractor": "visual_subject_extractor",
        "retriever": "retriever",
        "generator": "generator",
    }
)

# 4. Post-Visual Extraction Conditional Routing
workflow.add_conditional_edges(
    "visual_subject_extractor",
    post_visual_extraction_router,
    {
        "retriever": "retriever",
        "generator": "generator",
    }
)

# 5. Retriever Exit Conditional Edge Routing
workflow.add_conditional_edges(
    "retriever",
    knowledge_base_router,
    {
        "generator": "generator",
        "web_search_tool": "web_search_tool",
        "safe_refusal_node": "safe_refusal_node",
    }
)

# 6. Web Search Validation Edge Routing
workflow.add_conditional_edges(
    "web_search_tool",
    web_search_validation_router,
    {
        "generator": "generator",
        "safe_refusal_node": "safe_refusal_node",
    }
)

workflow.add_edge("safe_refusal_node", END)
workflow.add_edge("generator", END)

# Compile finalized execution application
app = workflow.compile()