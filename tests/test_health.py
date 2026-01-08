"""
Health endpoint tests.
"""

import pytest
from fastapi.testclient import TestClient

from clinicai.app import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def test_health_endpoint(client):
    """Test that the /health endpoint returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert data["data"]["status"] == "healthy"
    assert "timestamp" in data["data"]
    assert "version" in data["data"]
    assert "service" in data["data"]


def test_health_ready_endpoint(client):
    """Test that the /health/ready endpoint returns 200 OK."""
    response = client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data


def test_root_endpoint(client):
    """Test that the root endpoint returns API information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "version" in data
    assert "status" in data
    assert data["status"] == "running"
