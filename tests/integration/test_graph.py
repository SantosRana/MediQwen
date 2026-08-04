# tests/integration/test_graph.py
import pytest
from PIL import Image
from src.agent.graph import compile_workflow

@pytest.fixture
def compiled_graph():
    """Compiles the full active production LangGraph state framework."""
    return compile_workflow()

def test_end_to_end_emergency_multimodal_graph_flow(compiled_graph, tmp_path):
    """Verifies state data propagation across all nodes from guardrail check to generation output."""
    # 1. Setup a minimal test image asset
    mock_img_path = tmp_path / "emergency_chest_xray.jpg"
    Image.new("RGB", (600, 600), color="black").save(mock_img_path)

    # 2. Package real runtime dictionary inputs
    inputs = {
        "user_query": "I am gasping for air and my throat feels swollen shut!",
        "image_path": str(mock_img_path)
    }

    # 3. Execution Pass
    final_state = compiled_graph.invoke(inputs)

    # 4. Pipeline Assertions
    assert final_state is not None, "Graph execution returned None state."
    assert final_state.get("risk_level") == "EMERGENCY", f"Expected EMERGENCY risk level, got {final_state.get('risk_level')}"
    assert final_state.get("processed_image_payload") is not None, "Image payload was not processed."
    
    # Assert retrieved_context is returned as a list (whether empty or populated)
    assert isinstance(final_state.get("retrieved_context"), list), "retrieved_context must be a list."
    
    # Assert response text exists and matches the behavior spec formatting guidelines
    response_text = final_state.get("agent_response", "")
    assert len(response_text) > 0, "Agent response is empty."
    assert "🚨" in response_text or response_text.strip().startswith("🚨"), "Emergency response missing expected emergency icon indicator."