r"""
Tests for Mutual Predictability Scorer.

Based on the paper formula: MP(L) = Σ log P(y_i | x_i, L \ {(x_i, y_i)})
No API calls - uses mocks.
"""
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scorer import MutualPredictabilityScorer
from src.data_models import TruthfulQASample


# Mock helpers

def create_mock_client():
    """Create a mock HyperbolicClient."""
    return Mock()


def create_mock_logprob_response(token_logprobs_dict):
    """
    Create mock response with logprobs.

    Args:
        token_logprobs_dict: Dict mapping tokens to logprobs, e.g., {"True": -0.5, "False": -2.0}
    """
    response = Mock()

    # Create mock logprobs structure
    logprobs = Mock()

    # For base model, top_logprobs[0] is a dict: {token_str: logprob_float, ...}
    logprobs.top_logprobs = [token_logprobs_dict]

    return response, logprobs


# Tests

def test_compute_score_empty():
    """Test compute_score with empty list."""
    client = create_mock_client()
    scorer = MutualPredictabilityScorer(client)

    score = scorer.compute_score([])

    assert score == 0.0, "Empty list should return 0.0"


def test_compute_score_single_sample():
    """Test compute_score with single sample."""
    client = create_mock_client()
    scorer = MutualPredictabilityScorer(client)

    samples = [TruthfulQASample("Q1", "C1", predicted_label=True)]

    score = scorer.compute_score(samples)

    # With less than 2 samples, should return 0.0
    assert score == 0.0, "Single sample should return 0.0"


def test_get_label_logprob_true_label():
    """Test getting logprob for True label."""
    client = create_mock_client()
    scorer = MutualPredictabilityScorer(client)

    # Mock response with True having higher probability
    response, logprobs = create_mock_logprob_response({
        "True": -0.5,
        "False": -2.0
    })

    client.generate.return_value = response
    client.get_logprobs.return_value = logprobs

    sample = TruthfulQASample("Is sky blue?", "Sky is blue", predicted_label=True)
    other_samples = [TruthfulQASample("Is water wet?", "Water is wet", predicted_label=True)]

    logprob = scorer.get_label_logprob(sample, other_samples)

    assert logprob == -0.5, f"Expected -0.5 for True label, got {logprob}"


def test_get_logprob_for_label_with_specified_label():
    """Test getting logprob for a specified label (not the sample's label)."""
    client = create_mock_client()
    scorer = MutualPredictabilityScorer(client)

    response, logprobs = create_mock_logprob_response({
        "True": -1.5,
        "False": -0.8
    })

    client.generate.return_value = response
    client.get_logprobs.return_value = logprobs

    # Sample has predicted_label=True, but we ask for False
    sample = TruthfulQASample("Is earth flat?", "Earth is flat", predicted_label=True)
    other_samples = []

    # Get logprob for False (not the sample's actual label)
    logprob = scorer.get_logprob_for_label(sample, other_samples, label=False)

    assert logprob == -0.8, f"Expected -0.8 for False label, got {logprob}"


def test_get_logprob_for_label_matches_get_label_logprob():
    """Test that get_label_logprob is equivalent to get_logprob_for_label with sample.label."""
    client = create_mock_client()
    scorer = MutualPredictabilityScorer(client)

    response, logprobs = create_mock_logprob_response({
        "True": -1.2,
        "False": -2.5
    })

    client.generate.return_value = response
    client.get_logprobs.return_value = logprobs

    sample = TruthfulQASample("Test?", "Test claim", predicted_label=True)
    other_samples = []

    # Both methods should give same result
    logprob1 = scorer.get_label_logprob(sample, other_samples)

    # Reset mock for second call
    client.generate.return_value = response
    client.get_logprobs.return_value = logprobs

    logprob2 = scorer.get_logprob_for_label(sample, other_samples, sample.predicted_label)

    assert logprob1 == logprob2, \
        f"get_label_logprob ({logprob1}) should match get_logprob_for_label ({logprob2})"


def test_get_label_logprob_false_label():
    """Test getting logprob for False label."""
    client = create_mock_client()
    scorer = MutualPredictabilityScorer(client)

    response, logprobs = create_mock_logprob_response({
        "True": -2.0,
        "False": -0.5
    })

    client.generate.return_value = response
    client.get_logprobs.return_value = logprobs

    sample = TruthfulQASample("Is earth flat?", "Earth is flat", predicted_label=False)
    other_samples = [TruthfulQASample("Is sky blue?", "Sky is blue", predicted_label=True)]

    logprob = scorer.get_label_logprob(sample, other_samples)

    assert logprob == -0.5, f"Expected -0.5 for False label, got {logprob}"


def test_get_label_logprob_missing_token():
    """Test handling when target label token is not in top_logprobs."""
    client = create_mock_client()
    scorer = MutualPredictabilityScorer(client)

    # Response only has "Maybe" in logprobs, not "True" or "False"
    response, logprobs = create_mock_logprob_response({
        "Maybe": -1.0,
        "Unknown": -2.0
    })

    client.generate.return_value = response
    client.get_logprobs.return_value = logprobs

    sample = TruthfulQASample("Q?", "C", predicted_label=True)
    other_samples = [TruthfulQASample("Q2?", "C2", predicted_label=False)]

    logprob = scorer.get_label_logprob(sample, other_samples)

    assert logprob == float('-inf'), "Missing token should return -inf"


def test_get_label_logprob_no_logprobs():
    """Test handling when response has no logprobs."""
    client = create_mock_client()
    scorer = MutualPredictabilityScorer(client)

    response = Mock()
    logprobs = None

    client.generate.return_value = response
    client.get_logprobs.return_value = logprobs

    sample = TruthfulQASample("Q?", "C", predicted_label=True)
    other_samples = []

    logprob = scorer.get_label_logprob(sample, other_samples)

    assert logprob == float('-inf'), "No logprobs should return -inf"


def test_get_label_logprob_token_whitespace():
    """Test that token whitespace is handled correctly."""
    client = create_mock_client()
    scorer = MutualPredictabilityScorer(client)

    # Tokens may have leading/trailing whitespace
    response, logprobs = create_mock_logprob_response({
        " True ": -0.5,
        " False ": -2.0
    })

    client.generate.return_value = response
    client.get_logprobs.return_value = logprobs

    sample = TruthfulQASample("Q?", "C", predicted_label=True)
    other_samples = []

    logprob = scorer.get_label_logprob(sample, other_samples)

    # Should strip whitespace and match
    assert logprob == -0.5, "Should handle whitespace in tokens"


def test_compute_score_two_samples():
    """Test compute_score formula with two samples."""
    client = create_mock_client()
    scorer = MutualPredictabilityScorer(client)

    samples = [
        TruthfulQASample("Q1", "C1", predicted_label=True),
        TruthfulQASample("Q2", "C2", predicted_label=False)
    ]

    # Mock responses for each call
    response1, logprobs1 = create_mock_logprob_response({"True": -1.0, "False": -3.0})
    response2, logprobs2 = create_mock_logprob_response({"True": -3.5, "False": -0.5})

    client.generate.side_effect = [response1, response2]
    client.get_logprobs.side_effect = [logprobs1, logprobs2]

    score = scorer.compute_score(samples)

    # Score = log P(y1|x1, {(x2,y2)}) + log P(y2|x2, {(x1,y1)})
    # Score = -1.0 (True) + -0.5 (False) = -1.5
    expected = -1.0 + -0.5

    assert score == expected, f"Expected {expected}, got {score}"


def test_compute_score_three_samples():
    """Test compute_score follows paper formula with three samples."""
    client = create_mock_client()
    scorer = MutualPredictabilityScorer(client)

    samples = [
        TruthfulQASample("Q1", "C1", predicted_label=True),
        TruthfulQASample("Q2", "C2", predicted_label=True),
        TruthfulQASample("Q3", "C3", predicted_label=False)
    ]

    # Mock responses
    response1, logprobs1 = create_mock_logprob_response({"True": -0.8, "False": -2.0})
    response2, logprobs2 = create_mock_logprob_response({"True": -1.2, "False": -2.5})
    response3, logprobs3 = create_mock_logprob_response({"True": -3.0, "False": -0.3})

    client.generate.side_effect = [response1, response2, response3]
    client.get_logprobs.side_effect = [logprobs1, logprobs2, logprobs3]

    score = scorer.compute_score(samples)

    # MP(L) = Σ log P(y_i | x_i, L \ {(x_i, y_i)})
    # Score = -0.8 (True) + -1.2 (True) + -0.3 (False)
    expected = -0.8 + -1.2 + -0.3

    assert abs(score - expected) < 1e-9, f"Expected {expected}, got {score}"


def test_compute_score_calls_correct_context():
    """Test that compute_score passes correct context for each sample."""
    client = create_mock_client()
    scorer = MutualPredictabilityScorer(client)

    samples = [
        TruthfulQASample("Q1", "C1", predicted_label=True),
        TruthfulQASample("Q2", "C2", predicted_label=False),
        TruthfulQASample("Q3", "C3", predicted_label=True)
    ]

    # Mock responses
    response, logprobs = create_mock_logprob_response({"True": -1.0, "False": -1.0})
    client.generate.return_value = response
    client.get_logprobs.return_value = logprobs

    # Track calls to verify context
    calls_made = []
    original_get_logprob = scorer.get_label_logprob

    def track_calls(sample, other_samples):
        calls_made.append((sample, len(other_samples)))
        return original_get_logprob(sample, other_samples)

    scorer.get_label_logprob = track_calls

    scorer.compute_score(samples)

    # Should call get_label_logprob 3 times
    assert len(calls_made) == 3, f"Expected 3 calls, got {len(calls_made)}"

    # Each call should have n-1 context samples
    for i, (sample, context_size) in enumerate(calls_made):
        assert sample == samples[i], f"Wrong sample at index {i}"
        assert context_size == 2, f"Expected 2 context samples, got {context_size}"


def test_compute_score_excludes_target_from_context():
    """Test that target sample is excluded from its own context."""
    client = create_mock_client()
    scorer = MutualPredictabilityScorer(client)

    samples = [
        TruthfulQASample("Q1", "C1", predicted_label=True),
        TruthfulQASample("Q2", "C2", predicted_label=False)
    ]

    # Track which samples are used as context
    context_tracking = []

    def mock_format_icl(context_samples, query_sample):
        context_tracking.append({
            'query': query_sample.question,
            'context': [s.question for s in context_samples]
        })
        return "mock prompt"

    scorer.formatter.format_icl = mock_format_icl

    response, logprobs = create_mock_logprob_response({"True": -1.0, "False": -1.0})
    client.generate.return_value = response
    client.get_logprobs.return_value = logprobs

    scorer.compute_score(samples)

    # Check first call: Q1 as query, Q2 in context
    assert context_tracking[0]['query'] == "Q1"
    assert context_tracking[0]['context'] == ["Q2"]
    assert "Q1" not in context_tracking[0]['context'], "Target should not be in its own context"

    # Check second call: Q2 as query, Q1 in context
    assert context_tracking[1]['query'] == "Q2"
    assert context_tracking[1]['context'] == ["Q1"]
    assert "Q2" not in context_tracking[1]['context'], "Target should not be in its own context"


def test_compute_score_handles_inf_logprobs():
    """Test compute_score when some logprobs are -inf."""
    client = create_mock_client()
    scorer = MutualPredictabilityScorer(client)

    samples = [
        TruthfulQASample("Q1", "C1", predicted_label=True),
        TruthfulQASample("Q2", "C2", predicted_label=False)
    ]

    # First sample returns valid logprob, second returns -inf
    response1, logprobs1 = create_mock_logprob_response({"True": -1.5, "False": -2.0})
    response2 = Mock()
    logprobs2 = None  # Will return -inf

    client.generate.side_effect = [response1, response2]
    client.get_logprobs.side_effect = [logprobs1, logprobs2]

    score = scorer.compute_score(samples)

    # Score = -1.5 + (-inf) = -inf
    assert score == float('-inf'), f"Expected -inf when any logprob is -inf, got {score}"


def test_model_called_with_correct_params():
    """Test that model is called with correct parameters."""
    client = create_mock_client()
    scorer = MutualPredictabilityScorer(client)

    samples = [
        TruthfulQASample("Q1", "C1", predicted_label=True),
        TruthfulQASample("Q2", "C2", predicted_label=False)
    ]

    response, logprobs = create_mock_logprob_response({"True": -1.0, "False": -1.0})
    client.generate.return_value = response
    client.get_logprobs.return_value = logprobs

    scorer.compute_score(samples)

    # Check that generate() was called with correct params
    assert client.generate.call_count == 2, "Should call generate() twice for 2 samples"

    # Check parameters of calls
    for call in client.generate.call_args_list:
        args, kwargs = call
        # Check max_tokens=1, temperature=0.0, logprobs=5
        assert kwargs.get('max_tokens') == 1, "Should use max_tokens=1"
        assert kwargs.get('temperature') == 0.0, "Should use temperature=0.0"
        assert kwargs.get('logprobs') == 5, "Should request top 5 logprobs"


def test_paper_formula_correctness():
    """
    Test that implementation matches paper formula:
    MP(L) = Σ_{i=0}^{n} log p(y_i | x_i, L \\ {(x_i, y_i)})
    """
    client = create_mock_client()
    scorer = MutualPredictabilityScorer(client)

    print("\n=== Test: Paper Formula Correctness ===")

    samples = [
        TruthfulQASample("Q1", "C1", predicted_label=True),
        TruthfulQASample("Q2", "C2", predicted_label=True),
        TruthfulQASample("Q3", "C3", predicted_label=False)
    ]

    # Define specific logprobs for each sample
    logprob_values = [-0.5, -1.2, -0.8]

    responses_logprobs = [
        create_mock_logprob_response({"True": -0.5, "False": -3.0}),
        create_mock_logprob_response({"True": -1.2, "False": -2.5}),
        create_mock_logprob_response({"True": -3.5, "False": -0.8})
    ]

    client.generate.side_effect = [r for r, _ in responses_logprobs]
    client.get_logprobs.side_effect = [l for _, l in responses_logprobs]

    score = scorer.compute_score(samples)

    # Manual calculation according to paper formula
    expected = sum(logprob_values)

    print(f"Sample 1 (True): log P = {logprob_values[0]}")
    print(f"Sample 2 (True): log P = {logprob_values[1]}")
    print(f"Sample 3 (False): log P = {logprob_values[2]}")
    print(f"Total MP(L) = {expected}")
    print(f"Computed score = {score}")

    assert abs(score - expected) < 1e-9, \
        f"Score should match paper formula: expected {expected}, got {score}"


if __name__ == "__main__":
    print("Running scorer tests...\n")

    test_compute_score_empty()
    print("✓ Test: compute_score_empty")

    test_compute_score_single_sample()
    print("✓ Test: compute_score_single_sample")

    test_get_label_logprob_true_label()
    print("✓ Test: get_label_logprob_true_label")

    test_get_logprob_for_label_with_specified_label()
    print("✓ Test: get_logprob_for_label_with_specified_label")

    test_get_logprob_for_label_matches_get_label_logprob()
    print("✓ Test: get_logprob_for_label_matches_get_label_logprob")

    test_get_label_logprob_false_label()
    print("✓ Test: get_label_logprob_false_label")

    test_get_label_logprob_missing_token()
    print("✓ Test: get_label_logprob_missing_token")

    test_get_label_logprob_no_logprobs()
    print("✓ Test: get_label_logprob_no_logprobs")

    test_get_label_logprob_token_whitespace()
    print("✓ Test: get_label_logprob_token_whitespace")

    test_compute_score_two_samples()
    print("✓ Test: compute_score_two_samples")

    test_compute_score_three_samples()
    print("✓ Test: compute_score_three_samples")

    test_compute_score_calls_correct_context()
    print("✓ Test: compute_score_calls_correct_context")

    test_compute_score_excludes_target_from_context()
    print("✓ Test: compute_score_excludes_target_from_context")

    test_compute_score_handles_inf_logprobs()
    print("✓ Test: compute_score_handles_inf_logprobs")

    test_model_called_with_correct_params()
    print("✓ Test: model_called_with_correct_params")

    test_paper_formula_correctness()
    print("✓ Test: paper_formula_correctness")

    print("\n✅ All tests passed!")
