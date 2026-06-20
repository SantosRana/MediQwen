# src/agent/graph.py
from langgraph.graph import StateGraph, END
from .state import MedicalAgentState
from .nodes import (
    run_guardrail_check, 
    run_risk_classification, 
    run_multimodal_processing, 
    run_vector_search, 
    run_gemma_generation
)

# 1. Create StateGraph wrapper
workflow = StateGraph(MedicalAgentState)

# 2. Register functional operational nodes
workflow.add_node("guardrail", run_guardrail_check)
workflow.add_node("classifier", run_risk_classification)
workflow.add_node("multimodal_processor", run_multimodal_processing)
workflow.add_node("retriever", run_vector_search)
workflow.add_node("generator", run_gemma_generation)

# 3. Stitch edges based on the topology verified yesterday (Screenshot image_b3c448.png)
workflow.set_entry_point("guardrail")

# A) Route After Guardrail (based on policy safety flag)
def route_after_guardrail(state: MedicalAgentState):
    if not state["is_safe"]:
        print("🛑 Input flagged by safety policies. Routing directly to output termination.")
        return "generator"
    return "classifier"

workflow.add_conditional_edges("guardrail", route_after_guardrail)

# B) Route After Classifier (based on context tier gravity)
def route_after_classification(state: MedicalAgentState):
    if state["risk_level"] == "emergency":
        print("🚨 Severe urgency! routing straight to Generator for bypass escalation.")
        return "generator"
    return "multimodal_processor"

workflow.add_conditional_edges("classifier", route_after_classification)

# C) Construct sequential linkages for clean inquiries
workflow.add_edge("multimodal_processor", "retriever")
workflow.add_edge("retriever", "generator")
workflow.add_edge("generator", END)

# 4. Compile the graph configuration so it's ready for instantiation
app = workflow.compile()