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
    safe_refusal_node,
    run_qwen_generation
)

logger = logging.getLogger("graph_orchestrator")

# Instantiate LangGraph overall workflow builder
workflow = StateGraph(AgentState)

# Register All Concrete Working Processing Nodes
workflow.add_node("dialogue_manager", run_dialogue_manager)
workflow.add_node("classifier", run_risk_classification)
workflow.add_node("multimodal_processor", run_multimodal_processing)
workflow.add_node("retriever", run_vector_search)
workflow.add_node("web_search_tool", run_web_search_tool)
workflow.add_node("safe_refusal_node", safe_refusal_node)
workflow.add_node("generator", run_qwen_generation)


# =======================================================================
# 🌐 CONDITIONAL ROUTING FUNCTIONS
# =======================================================================

def dialogue_fsm_switchboard(state: AgentState) -> str:
    """
    Evaluates primary state machine status.
    Routes blocked queries directly to refusal, and all authorized interactions
    to risk classification and pipeline processing.
    """
    active_state = state.get("dialog_state", "chat").lower().strip()
    
    if active_state == "blocked":
        return "blocked_refusal"
        
    logger.info(f"🧬 FSM Switchboard Execution: Routing via state [{active_state.upper()}] to Risk Classifier.")
    return "authorized_pipeline"


def multimodal_image_router(state: AgentState) -> str:
    """
    Checks if an authorized image payload exists before running the preprocessor.
    """
    trace = state.get("routing_trace", [])
    process_image = bool(state.get("process_image", False))
    
    if process_image:
        state["routing_trace"] = trace + ["classifier -> multimodal_processor"]
        return "multimodal_processor"
    
    state["routing_trace"] = trace + ["classifier -> text_only_retriever"]
    return "retriever"


def knowledge_base_router(state: AgentState) -> str:
    """
    Evaluates intent-driven retrieval status and enforces evidence grounding.
    - If retrieval is NOT requested (e.g., Turn 1 visual triage or casual chat),
      routes directly to generator.
    - If retrieval IS requested but local DB returns 0 chunks and offline,
      routes directly to safe_refusal_node to prevent hallucinated answers.
    """
    trace = state.get("routing_trace", [])
    requires_retrieval = state.get("requires_retrieval", False)
    context = state.get("retrieved_context", [])
    is_emergency = state.get("risk_level", "low").lower() == "emergency"
    is_online = state.get("is_online", False)

    # 1. Deferred RAG Short-Circuit: Skip retrieval if user did not ask for clinical context
    if not requires_retrieval:
        logger.info("⏩ Retrieval not requested for this turn. Short-circuiting directly to Generator.")
        state["routing_trace"] = trace + ["retriever -> no_retrieval_needed -> generator"]
        return "generator"

    # 2. Context Found: Proceed to grounded generation
    if context:
        logger.info(f"✅ RAG Context Located ({len(context)} chunks). Routing to Generator.")
        state["routing_trace"] = trace + ["retriever -> context_found -> generator"]
        return "generator"

    # 3. Emergency Fallback: Prioritize immediate safety response over missing DB
    if is_emergency:
        logger.warning("🚨 EMERGENCY DATABASE MISS: Routing directly to Generator to deliver safety warnings.")
        state["routing_trace"] = trace + ["retriever -> emergency_db_miss -> generator"]
        return "generator"

    # 4. DB Miss Handling: Route to Web Search if online, otherwise enforce Safe Refusal
    if is_online:
        logger.info("🌐 Local DB miss. Web fallback enabled. Routing to web_search_tool.")
        state["routing_trace"] = trace + ["retriever -> db_miss_online -> web_search_tool"]
        return "web_search_tool"
    else:
        logger.warning("🛡️ Local DB miss in Offline Mode. Routing to safe_refusal_node to prevent ungrounded generation.")
        state["routing_trace"] = trace + ["retriever -> db_miss_offline -> safe_refusal_node"]
        return "safe_refusal_node"


def web_search_validation_router(state: AgentState) -> str:
    """
    Verifies if the web scraper found any verified context chunks before sending to generator.
    """
    trace = state.get("routing_trace", [])
    context = state.get("retrieved_context", [])
    
    if context:
        state["routing_trace"] = trace + ["web_search_tool -> web_context_found -> generator"]
        return "generator"
        
    logger.error("❌ Retry Path Exhausted: Web searches returned zero verified documents.")
    state["routing_trace"] = trace + ["web_search_tool -> empty_web_miss -> safe_refusal_node"]
    return "safe_refusal_node"


# =======================================================================
# 🔗 GRAPH EDGE BINDINGS
# =======================================================================

workflow.add_edge(START, "dialogue_manager")

# Node 1 Master Switchboard Edge Routing
workflow.add_conditional_edges(
    "dialogue_manager", 
    dialogue_fsm_switchboard,
    {
        "blocked_refusal": "safe_refusal_node", 
        "authorized_pipeline": "classifier"
    }
)

# Node 2 Exit Conditional Edge Routing
workflow.add_conditional_edges(
    "classifier",
    multimodal_image_router,
    {
        "multimodal_processor": "multimodal_processor",
        "retriever": "retriever"
    }
)

# Node 3 Exit Edge Routing (Direct to Retriever)
workflow.add_edge("multimodal_processor", "retriever")

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

# Compile finalized execution application
app = workflow.compile()