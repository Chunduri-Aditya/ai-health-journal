"""Tests for /prompt endpoint."""

import pytest
from unittest.mock import patch, MagicMock
from app import app


@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@patch('app.requests.post')
def test_prompt_success(mock_post, client):
    """Test /prompt success path with mocked Ollama."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "What are you grateful for today?"}
    mock_post.return_value = mock_response
    
    response = client.post('/prompt')
    assert response.status_code == 200
    data = response.get_json()
    assert "prompt" in data
    assert len(data["prompt"]) > 0
    assert data["prompt"] == "What are you grateful for today?"


@patch('app.requests.post')
def test_prompt_ollama_error(mock_post, client):
    """Test /prompt handles Ollama error."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_post.return_value = mock_response
    
    response = client.post('/prompt')
    assert response.status_code == 500
    data = response.get_json()
    assert "error" in data


@patch('app.requests.post')
def test_prompt_ollama_exception(mock_post, client):
    """Test /prompt handles Ollama exception."""
    mock_post.side_effect = Exception("Connection error")
    
    response = client.post('/prompt')
    assert response.status_code == 500
    data = response.get_json()
    assert "error" in data
