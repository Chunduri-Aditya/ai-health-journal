"""Tests for quality pipeline (Draft → Verify → Revise)."""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from llm_client import json_generate, ollama_generate, check_ollama_available


@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    app.config['SECRET_KEY'] = 'test-secret-key-for-sessions'
    with app.test_client() as client:
        yield client


@patch('app.check_ollama_available')
@patch('app.json_generate')
def test_analyze_quality_mode_returns_json_schema(mock_json_gen, mock_check, client):
    """Test /analyze with quality_mode returns valid JSON schema."""
    mock_check.return_value = True
    
    # Mock draft generation
    draft_json = {
        "summary": "Test summary",
        "emotions": ["anxiety", "stress"],
        "patterns": ["avoidance"],
        "triggers": ["work"],
        "coping_suggestions": ["Take breaks", "Practice mindfulness"],
        "quotes_from_user": ["I feel stressed"],
        "confidence": 0.8
    }
    
    # Mock verifier (passes verification)
    verdict = {
        "groundedness_score": 0.9,
        "unsupported_claims": [],
        "safety_flags": [],
        "rewrite_required": False,
        "rewrite_instructions": ""
    }
    
    mock_json_gen.side_effect = [draft_json, verdict]
    
    response = client.post('/analyze', json={
        "entry": "I feel stressed about work.",
        "quality_mode": True
    })
    
    assert response.status_code == 200
    data = response.get_json()
    assert "insight" in data
    assert "analysis" in data
    
    # Verify analysis has required fields
    analysis = data["analysis"]
    assert "summary" in analysis
    assert "emotions" in analysis
    assert "coping_suggestions" in analysis
    assert "confidence" in analysis


@patch('app.check_ollama_available')
@patch('app.json_generate')
def test_verifier_triggers_revision(mock_json_gen, mock_check, client):
    """Test that verifier triggers revision when draft has issues."""
    mock_check.return_value = True
    
    # Mock draft with unsupported claims
    draft_json = {
        "summary": "You've been stressed about work for months",
        "emotions": ["anxiety"],
        "patterns": [],
        "triggers": [],
        "coping_suggestions": ["Take breaks"],
        "quotes_from_user": [],
        "confidence": 0.9
    }
    
    # Mock verifier (flags issues - requires rewrite)
    verdict = {
        "groundedness_score": 0.5,
        "unsupported_claims": ["'for months' is not supported by entry"],
        "safety_flags": [],
        "rewrite_required": True,
        "rewrite_instructions": "Remove temporal claims not in entry"
    }
    
    # Mock revision
    revised_json = {
        "summary": "You're feeling stressed about work",
        "emotions": ["anxiety"],
        "patterns": [],
        "triggers": [],
        "coping_suggestions": ["Take breaks"],
        "quotes_from_user": [],
        "confidence": 0.8
    }
    
    mock_json_gen.side_effect = [draft_json, verdict, revised_json]
    
    response = client.post('/analyze', json={
        "entry": "I feel stressed about work.",
        "quality_mode": True
    })
    
    assert response.status_code == 200
    # Should have called json_generate 3 times (draft, verify, revise)
    assert mock_json_gen.call_count == 3


@patch('rag_store.get_rag_store')
@patch('llm_client.check_ollama_available')
@patch('llm_client.ollama_generate')
def test_retrieval_returns_empty_when_disabled(mock_gen, mock_check, mock_get_rag, client):
    """Test that retrieval returns empty gracefully when store disabled."""
    mock_check.return_value = True
    mock_gen.return_value = "Test insight"
    
    # Mock disabled RAG store
    mock_rag_instance = MagicMock()
    mock_rag_instance.enabled = False
    mock_rag_instance.retrieve.return_value = ""
    mock_get_rag.return_value = mock_rag_instance
    
    response = client.post('/analyze', json={
        "entry": "Test entry",
        "quality_mode": False
    })
    
    assert response.status_code == 200
    # Should not call retrieve when disabled
    mock_rag_instance.retrieve.assert_not_called()


@patch('app.check_ollama_available')
def test_models_endpoint_returns_config(mock_check, client):
    """Test /models endpoint returns model configuration."""
    mock_check.return_value = True
    
    response = client.get('/models')
    assert response.status_code == 200
    data = response.get_json()
    
    assert "generator" in data
    assert "fallback" in data
    assert "verifier" in data
    assert "prompt" in data
    assert "quality_mode_default" in data
    assert "retrieval_enabled" in data
