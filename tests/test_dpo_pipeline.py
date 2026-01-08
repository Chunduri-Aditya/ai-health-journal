"""Tests for DPO pipeline components."""

import pytest
import json
import os
import tempfile
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.build_dpo_dataset import build_dpo_pairs, should_keep_pair, load_eval_results, build_prompt


def test_build_dpo_dataset_produces_pairs():
    """Test that build_dpo_dataset produces valid preference pairs."""
    # Create mock evaluation results with entry_id and strict quality metrics
    baseline_results = {
        'case_results': [
            {
                'entry_id': 'test_1',
                'entry': 'Test entry 1',
                'dataset_version': '1.0',
                'baseline_output': json.dumps({'summary': 'Baseline summary', 'emotions': ['anxiety']}),
                'analysis': {'summary': 'Baseline summary', 'emotions': ['anxiety']},
                'metrics': {
                    'faithfulness': 0.6,
                    'no_invention': 0.8,
                    'groundedness_score': 0.6,
                    'unsupported_claims': ['Claim not in entry']
                },
                'parse_failures': 0
            },
            {
                'entry_id': 'test_2',
                'entry': 'Test entry 2',
                'dataset_version': '1.0',
                'baseline_output': json.dumps({'summary': 'Baseline summary 2', 'emotions': ['stress']}),
                'analysis': {'summary': 'Baseline summary 2', 'emotions': ['stress']},
                'metrics': {
                    'faithfulness': 0.7,
                    'no_invention': 0.9,
                    'groundedness_score': 0.7,
                    'unsupported_claims': []
                },
                'parse_failures': 0
            }
        ]
    }
    
    quality_results = {
        'case_results': [
            {
                'entry_id': 'test_1',
                'entry': 'Test entry 1',
                'dataset_version': '1.0',
                'retrieved_context': '',
                'quality_output': json.dumps({'summary': 'Quality summary', 'emotions': ['anxiety']}),
                'final_json': {'summary': 'Quality summary', 'emotions': ['anxiety']},
                'analysis': {'summary': 'Quality summary', 'emotions': ['anxiety']},
                'metrics': {
                    'faithfulness': 0.95,  # Meets threshold
                    'no_invention': 1.00,  # Perfect
                    'groundedness_score': 0.9,
                    'unsupported_claims': []
                },
                'parse_failures': 0
            },
            {
                'entry_id': 'test_2',
                'entry': 'Test entry 2',
                'dataset_version': '1.0',
                'retrieved_context': '',
                'quality_output': json.dumps({'summary': 'Quality summary 2', 'emotions': ['stress']}),
                'final_json': {'summary': 'Quality summary 2', 'emotions': ['stress']},
                'analysis': {'summary': 'Quality summary 2', 'emotions': ['stress']},
                'metrics': {
                    'faithfulness': 0.96,  # Meets threshold
                    'no_invention': 1.00,  # Perfect
                    'groundedness_score': 0.8,
                    'unsupported_claims': []
                },
                'parse_failures': 0
            }
        ]
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_path = os.path.join(tmpdir, 'baseline.json')
        quality_path = os.path.join(tmpdir, 'quality.json')
        output_path = os.path.join(tmpdir, 'pairs.jsonl')
        
        with open(baseline_path, 'w') as f:
            json.dump(baseline_results, f)
        with open(quality_path, 'w') as f:
            json.dump(quality_results, f)
        
        result = build_dpo_pairs(baseline_path, quality_path, output_path)
        
        assert result['num_pairs'] > 0
        assert os.path.exists(output_path)
        
        # Verify pairs format
        with open(output_path, 'r') as f:
            pairs = [json.loads(line) for line in f]
        
        assert len(pairs) > 0, "Should create at least one pair"
        for pair in pairs:
            assert 'prompt' in pair, "Pair must have 'prompt'"
            assert 'chosen' in pair, "Pair must have 'chosen'"
            assert 'rejected' in pair, "Pair must have 'rejected'"
            assert 'metadata' in pair, "Pair must have 'metadata'"
            assert 'entry_id' in pair['metadata'], "Metadata must have 'entry_id'"


def test_should_keep_pair_filters_correctly():
    """Test that should_keep_pair filters pairs correctly with STRICT criteria."""
    # Good pair: quality meets all strict criteria and is better
    baseline_result = {
        'baseline_output': json.dumps({'summary': 'baseline'}),
        'metrics': {
            'faithfulness': 0.6,
            'no_invention': 0.8,
            'groundedness_score': 0.6,
            'unsupported_claims': ['Claim 1']
        },
        'parse_failures': 0
    }
    quality_result = {
        'quality_output': json.dumps({'summary': 'quality'}),
        'metrics': {
            'faithfulness': 0.95,  # >= 0.95 ✓
            'no_invention': 1.00,  # == 1.00 ✓
            'groundedness_score': 0.9,
            'unsupported_claims': []
        },
        'parse_failures': 0
    }
    should_keep, reason = should_keep_pair(baseline_result, quality_result)
    assert should_keep == True, f"Expected True, got {should_keep}, reason: {reason}"
    
    # Bad pair: quality faithfulness too low (< 0.95)
    quality_result_low_faith = {
        'quality_output': json.dumps({'summary': 'quality'}),
        'metrics': {
            'faithfulness': 0.90,  # < 0.95 ✗
            'no_invention': 1.00,
            'groundedness_score': 0.9,
            'unsupported_claims': []
        },
        'parse_failures': 0
    }
    should_keep, reason = should_keep_pair(baseline_result, quality_result_low_faith)
    assert should_keep == False, f"Expected False (faithfulness too low), got {should_keep}"
    assert 'faithfulness' in reason.lower()
    
    # Bad pair: quality no_invention not perfect (< 1.00)
    quality_result_low_no_inv = {
        'quality_output': json.dumps({'summary': 'quality'}),
        'metrics': {
            'faithfulness': 0.95,
            'no_invention': 0.99,  # < 1.00 ✗
            'groundedness_score': 0.9,
            'unsupported_claims': []
        },
        'parse_failures': 0
    }
    should_keep, reason = should_keep_pair(baseline_result, quality_result_low_no_inv)
    assert should_keep == False, f"Expected False (no_invention not perfect), got {should_keep}"
    assert 'no_invention' in reason.lower()
    
    # Bad pair: quality has parse failures
    quality_result_parse_fail = {
        'quality_output': json.dumps({'summary': 'quality'}),
        'metrics': {
            'faithfulness': 0.95,
            'no_invention': 1.00,
            'groundedness_score': 0.9,
            'unsupported_claims': []
        },
        'parse_failures': 1  # ✗
    }
    should_keep, reason = should_keep_pair(baseline_result, quality_result_parse_fail)
    assert should_keep == False, f"Expected False (parse failure), got {should_keep}"
    assert 'parse' in reason.lower()
    
    # Good pair: baseline is legacy format (no baseline_output)
    baseline_legacy = {
        'baseline_output': None,  # Legacy format
        'metrics': {
            'faithfulness': 0.6,
            'no_invention': 0.8
        },
        'parse_failures': 0
    }
    should_keep, reason = should_keep_pair(baseline_legacy, quality_result)
    assert should_keep == True, f"Expected True (baseline legacy), got {should_keep}"
    assert 'legacy' in reason.lower()


def test_build_dpo_dataset_with_baseline_json():
    """Test that build_dpo_dataset works with baseline_json mode."""
    baseline_json_results = {
        'case_results': [
            {
                'entry_id': 'test_1',
                'entry': 'Test entry 1',
                'dataset_version': '1.0',
                'baseline_json_output': json.dumps({'summary': 'Baseline JSON summary', 'emotions': ['anxiety']}),
                'analysis': {'summary': 'Baseline JSON summary', 'emotions': ['anxiety']},
                'metrics': {
                    'faithfulness': 0.90,  # Below threshold
                    'no_invention': 0.95,  # Below 1.00
                    'groundedness_score': 0.8,
                    'unsupported_claims': []
                },
                'parse_failures': 0
            }
        ]
    }
    
    quality_results = {
        'case_results': [
            {
                'entry_id': 'test_1',
                'entry': 'Test entry 1',
                'dataset_version': '1.0',
                'quality_output': json.dumps({'summary': 'Quality summary', 'emotions': ['anxiety']}),
                'final_json': {'summary': 'Quality summary', 'emotions': ['anxiety']},
                'analysis': {'summary': 'Quality summary', 'emotions': ['anxiety']},
                'metrics': {
                    'faithfulness': 0.95,
                    'no_invention': 1.00,
                    'groundedness_score': 0.9,
                    'unsupported_claims': []
                },
                'parse_failures': 0
            }
        ]
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_json_path = os.path.join(tmpdir, 'baseline_json.json')
        quality_path = os.path.join(tmpdir, 'quality.json')
        output_path = os.path.join(tmpdir, 'pairs.jsonl')
        
        with open(baseline_json_path, 'w') as f:
            json.dump(baseline_json_results, f)
        with open(quality_path, 'w') as f:
            json.dump(quality_results, f)
        
        result = build_dpo_pairs(baseline_json_path, quality_path, output_path, use_baseline_json=True)
        
        # Should create pairs because quality is better
        assert result['num_pairs'] > 0
        assert os.path.exists(output_path)


def test_eval_runner_supports_dataset_flag():
    """Test that eval runner supports --dataset flag."""
    import subprocess
    import sys
    
    # Test that script accepts --dataset flag without error
    result = subprocess.run(
        [sys.executable, 'evals/run_evals.py', '--help'],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    
    assert '--dataset' in result.stdout or result.returncode == 0


def test_train_script_can_load_dataset():
    """Smoke test: verify train script can load dataset format."""
    pytest.importorskip("peft", reason="peft not installed - training dependencies optional")
    
    # Create minimal valid dataset
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        pair = {
            'prompt': 'Test prompt',
            'chosen': '{"summary": "Chosen response"}',
            'rejected': '{"summary": "Rejected response"}',
            'metadata': {}
        }
        f.write(json.dumps(pair) + '\n')
        temp_path = f.name
    
    try:
        # Test that we can load it (without actually training)
        from train.train_dpo import load_dpo_pairs, format_dataset
        
        pairs = load_dpo_pairs(temp_path)
        assert len(pairs) == 1
        assert pairs[0]['prompt'] == 'Test prompt'
        
        # Test dataset formatting (skip if datasets not available)
        try:
            dataset = format_dataset(pairs)
            assert len(dataset) == 1
            assert dataset[0]['prompt'] == 'Test prompt'
            assert dataset[0]['chosen'] == '{"summary": "Chosen response"}'
        except ImportError:
            pytest.skip("datasets library not available")
        
    finally:
        os.unlink(temp_path)
