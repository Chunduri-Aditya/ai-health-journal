"""Tests for /ping endpoint."""

import pytest
from app import app


@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_ping_returns_ok(client):
    """Test /ping returns correct status."""
    response = client.get('/ping')
    assert response.status_code == 200
    data = response.get_json()
    assert data == {"status": "ok"}
