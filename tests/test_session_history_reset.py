"""Tests for /session/history and /session/reset endpoints."""

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


@patch('app.requests.get')
@patch('app.requests.post')
def test_session_history_empty(mock_post, mock_get, client):
    """Test /session/history returns empty list when no entries."""
    mock_get.return_value.status_code = 200
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "Test insight"}
    mock_post.return_value = mock_response
    
    response = client.get('/session/history')
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, list)
    assert len(data) == 0


@patch('app.requests.get')
@patch('app.requests.post')
def test_session_history_returns_entries(mock_post, mock_get, client):
    """Test /session/history returns entries after analysis."""
    mock_get_response = MagicMock()
    mock_get_response.status_code = 200
    mock_get.return_value = mock_get_response
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "Test insight"}
    mock_post.return_value = mock_response
    
    # Submit entry and verify it's stored and returned by endpoint
    response1 = client.post('/analyze', json={"entry": "Test entry"})
    assert response1.status_code == 200
    
    # Verify endpoint returns the entry
    response = client.get('/session/history')
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["entry"] == "Test entry"
    assert "response" in data[0]
    assert data[0]["response"] == "Test insight"
    
    # Verify endpoint structure is correct (list of dicts with entry/response keys)
    assert all(key in data[0] for key in ["entry", "response"])


@patch('app.requests.get')
@patch('app.requests.post')
def test_session_reset_clears_history(mock_post, mock_get, client):
    """Test /session/reset clears session history."""
    mock_get_response = MagicMock()
    mock_get_response.status_code = 200
    mock_get.return_value = mock_get_response
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "Test insight"}
    mock_post.return_value = mock_response
    
    # Add entry
    client.post('/analyze', json={"entry": "Test entry"})
    
    # Verify endpoint returns entry
    response = client.get('/session/history')
    assert len(response.get_json()) == 1
    
    # Reset session
    response = client.post('/session/reset')
    assert response.status_code == 200
    data = response.get_json()
    assert data == {"status": "cleared"}
    
    # Verify endpoint returns empty
    response = client.get('/session/history')
    assert len(response.get_json()) == 0


@patch('app.requests.get')
@patch('app.requests.post')
def test_session_persists_across_requests(mock_post, mock_get, client):
    """Test session persists across multiple requests."""
    mock_get_response = MagicMock()
    mock_get_response.status_code = 200
    mock_get.return_value = mock_get_response
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "Test insight"}
    mock_post.return_value = mock_response
    
    # Add entry
    client.post('/analyze', json={"entry": "Persistent entry"})
    
    # Get history (first request)
    response1 = client.get('/session/history')
    data1 = response1.get_json()
    assert len(data1) == 1
    entry_text = data1[0]["entry"]
    assert entry_text == "Persistent entry"
    
    # Get history again (second request, same session)
    response2 = client.get('/session/history')
    data2 = response2.get_json()
    assert len(data2) == 1
    assert data1[0]["entry"] == data2[0]["entry"]
