# src/agent/graph.py
"""
Production Orchestrator Graph for MediGemma.
Enforces non-linear, data-driven execution routing, strict validation gates,
and provides clear tracking arrays for debugging.
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
    run_safe_refusal_fallback,
    run_qwen_generation,
    run_conversational_skill_node
)

logger = logging.getLogger("graph_orchestrator")

# Instantiate LangGraph overall workflow builder
workflow = StateGraph(AgentState)

# Register All Concrete Working Processing Nodes
workflow.add_node("dialogue_manager", run_dialogue_manager)
workflow.add_node("conversational_skill_node", run_conversational_skill_node)
workflow.add_node("classifier", run_risk_classification)
workflow.add_node("multimodal_processor", run_multimodal_processing)
workflow.add_node("retriever", run_vector_search)
workflow.add_node("web_search_tool", run_web_search_tool)
workflow.add_node("safe_refusal_node", run_safe_refusal_fallback)
workflow.add_node("generator", run_qwen_generation)


def dialogue_fsm_switchboard(state: AgentState) -> str:
    """
    Evaluates state machine properties. 
    Routes closing frames through the clinical generator path to keep context intact.
    """
    active_state = state.get("dialog_state", "chat").lower().strip()
    
    if active_state == "blocked":
        return "blocked_refusal"
    elif active_state == "chat":
        return "casual_chat"
        
    # Default to clinical path for any other state (e.g., "clinical", "medical", etc.)
    logger.info(f"🧬 FSM Switchboard Execution: Routing via state [{active_state.upper()}] to Clinical Pipeline.")
    return "clinical_medical"


def multimodal_image_router(state: AgentState) -> str:
    """
    Saves an execution loop by checking if an image exists before running the preprocessor.
    """
    trace = state.get("routing_trace", [])
    image_present = bool(state.get("image_path"))
    
    if image_present:
        state["routing_trace"] = trace + ["classifier -> multimodal_processor"]
        return "multimodal_processor"
    
    state["routing_trace"] = trace + ["classifier -> text_only_retriever"]
    return "retriever"


def image_query_skip_retrieval_router(state: AgentState) -> str:
    """
    Context Awareness Optimization.
    Bypasses empty vector lookups if the query is an image-only question.
    """
    trace = state.get("routing_trace", [])
    query_text = state.get("user_query", "").lower().strip()
    
    # Detect generic visual questions ("what is this?", "look at this")
    IMAGE_ONLY_PROMPTS = {
        "what is this", "what is this?", "look at this", "look", 
        "can you check this", "check this image", "what does this show"
    }
    
    if query_text in IMAGE_ONLY_PROMPTS and state.get("processed_image_payload"):
        logger.warning("🖼️ Image-only interaction pattern detected. Bypassing vector search loops.")
        state["routing_trace"] = trace + ["multimodal_processor -> image_only_generator_shortcut"]
        return "generator"
        
    state["routing_trace"] = trace + ["multimodal_processor -> standard_vector_retriever"]
    return "retriever"


def knowledge_base_router(state: AgentState) -> str:
    """
    Evaluates document retrieval state.
    If local database records drop, checks if device can route to live web searches.
    """
    trace = state.get("routing_trace", [])
    context = state.get("retrieved_context", [])
    is_emergency = state.get("risk_level", "low").lower() == "emergency"
    
    if context:
        state["routing_trace"] = trace + ["retriever -> context_found -> generator"]
        return "generator"
        
    if is_emergency:
        logger.warning("🚨 EMERGENCY DATABASE MISS: Short-circuiting directly to Generator to reduce latency.")
        state["routing_trace"] = trace + ["retriever -> emergency_db_miss -> generator"]
        return "generator"
        
    if state.get("is_online") is True:
        state["routing_trace"] = trace + ["retriever -> db_miss -> web_search_tool"]
        return "web_search_tool"
    else:
        state["routing_trace"] = trace + ["retriever -> db_miss_offline -> safe_refusal_node"]
        return "safe_refusal_node"


def web_search_validation_router(state: AgentState) -> str:
    """
    Retry Verification Gate.
    Verifies if the web scraper found any context chunks before sending to the generator.
    """
    trace = state.get("routing_trace", [])
    context = state.get("retrieved_context", [])
    
    if context:
        state["routing_trace"] = trace + ["web_search_tool -> web_context_found -> generator"]
        return "generator"
        
    logger.error("❌ Retry Path Exhausted: Web searches returned zero verified documents.")
    state["routing_trace"] = trace + ["web_search_tool -> empty_web_miss -> safe_refusal_node"]
    return "safe_refusal_node"

# GRAPH EDGE BINDINGS
# ---------------------------------------------------------------------------
workflow.add_edge(START, "dialogue_manager")

# Node 1 Master Switchboard Edge Routing
workflow.add_conditional_edges(
    "dialogue_manager", 
    dialogue_fsm_switchboard,
    {
        "blocked_refusal": "safe_refusal_node", 
        "casual_chat": "conversational_skill_node",
        "clinical_medical": "classifier"
    }
)

workflow.add_edge("conversational_skill_node", END)

# Node 2 Exit Conditional Edge Routing
workflow.add_conditional_edges(
    "classifier",
    multimodal_image_router,
    {
        "multimodal_processor": "multimodal_processor",
        "retriever": "retriever"
    }
)

# Node 3 Exit Conditional Edge Routing
workflow.add_conditional_edges(
    "multimodal_processor",
    image_query_skip_retrieval_router,
    {
        "generator": "generator",
        "retriever": "retriever"
    }
)

# Node 4 Exit Conditional Edge Routing
workflow.add_conditional_edges(
    "retriever",
    knowledge_base_router,
    {
        "generator": "generator",
        "web_search_tool": "web_search_tool",
        "safe_refusal_node": "safe_refusal_node"
    }
)

# Node 5 Exit Conditional Edge Routing
workflow.add_conditional_edges(
    "web_search_tool",
    web_search_validation_router,
    {
        "generator": "generator",
        "safe_refusal_node": "safe_refusal_node"
    }
)

workflow.add_edge("safe_refusal_node", END)
workflow.add_edge("generator", END)

app = workflow.compile()