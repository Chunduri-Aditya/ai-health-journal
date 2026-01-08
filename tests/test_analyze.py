"""Tests for /analyze endpoint."""

import pytest
from unittest.mock import patch, MagicMock
from app import app


@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    app.config['SECRET_KEY'] = 'test-secret-key-for-sessions'
    with app.test_client() as client:
        yield client


def test_analyze_rejects_empty_entry(client):
    """Test /analyze rejects empty entry."""
    response = client.post('/analyze', json={"entry": ""})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert "Please enter" in data["error"]


def test_analyze_rejects_whitespace_only(client):
    """Test /analyze rejects whitespace-only entry."""
    response = client.post('/analyze', json={"entry": "   "})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_analyze_rejects_too_long_entry(client):
    """Test /analyze rejects entry > 1000 characters."""
    long_entry = "a" * 1001
    response = client.post('/analyze', json={"entry": long_entry})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert "too long" in data["error"].lower() or "limit" in data["error"].lower()


@patch('app.requests.get')
@patch('app.requests.post')
def test_analyze_success(mock_post, mock_get, client):
    """Test /analyze success path with mocked Ollama."""
    # Mock Ollama health check
    mock_get_response = MagicMock()
    mock_get_response.status_code = 200
    mock_get.return_value = mock_get_response
    
    # Mock Ollama generate response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "This is a test insight about your journal entry."}
    mock_post.return_value = mock_response
    
    response = client.post('/analyze', json={"entry": "I feel stressed today."})
    assert response.status_code == 200
    data = response.get_json()
    assert "insight" in data
    assert len(data["insight"]) > 0
    
    # Verify session was updated by checking /session/history endpoint
    history_response = client.get('/session/history')
    assert history_response.status_code == 200
    history_data = history_response.get_json()
    assert len(history_data) == 1
    assert history_data[0]["entry"] == "I feel stressed today."
    assert "response" in history_data[0]


@patch('app.requests.get')
def test_analyze_ollama_offline(mock_get, client):
    """Test /analyze handles Ollama offline gracefully."""
    from requests.exceptions import RequestException
    mock_get.side_effect = RequestException("Connection refused")
    
    response = client.post('/analyze', json={"entry": "Test entry."})
    assert response.status_code == 503
    data = response.get_json()
    assert "error" in data
    assert "offline" in data["error"].lower()


@patch('app.requests.get')
@patch('app.requests.post')
def test_analyze_ollama_error(mock_post, mock_get, client):
    """Test /analyze handles Ollama API error."""
    mock_get_response = MagicMock()
    mock_get_response.status_code = 200
    mock_get.return_value = mock_get_response
    
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_post.return_value = mock_response
    
    response = client.post('/analyze', json={"entry": "Test entry."})
    assert response.status_code == 500
    data = response.get_json()
    assert "error" in data
