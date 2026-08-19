# tests/unit/test_api_server.py
import pytest
from fastapi.testclient import TestClient
from app.api_server import server as app

client = TestClient(app)

def test_health_check_endpoint():
    """Verifies that the /health endpoint returns HTTP 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200

def test_chat_endpoint_validation():
    """Verifies that missing form data returns a 422 Validation Error."""
    response = client.post("/chat", data={})
    assert response.status_code == 422