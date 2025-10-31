"""
Minimal tests for PromptFormatter without API calls.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_models import TruthfulQASample
from src.prompt_formatter import PromptFormatter


def test_format_single_without_label():
    """Test formatting a single sample without label."""
    formatter = PromptFormatter()
    sample = TruthfulQASample(
        question="What is the capital of France?",
        choice="Paris is the capital of France",
        predicted_label=True
    )

    result = formatter.format_single(sample, include_label=False)

    print("\n=== Test: Single sample without label ===")
    print(result)

    assert "Question: What is the capital of France?" in result
    assert "Claim: Paris is the capital of France" in result
    assert "I think this Claim is" in result
    assert "True" not in result
    assert "False" not in result


def test_format_single_with_true_label():
    """Test formatting a single sample with True label."""
    formatter = PromptFormatter()
    sample = TruthfulQASample(
        question="Is the sky blue?",
        choice="The sky is blue",
        predicted_label=True
    )

    result = formatter.format_single(sample, include_label=True)

    print("\n=== Test: Single sample with True label ===")
    print(result)

    assert "Question: Is the sky blue?" in result
    assert "Claim: The sky is blue" in result
    assert "I think this Claim is True" in result


def test_format_single_with_false_label():
    """Test formatting a single sample with False label."""
    formatter = PromptFormatter()
    sample = TruthfulQASample(
        question="Is the Earth flat?",
        choice="The Earth is flat",
        predicted_label=False
    )

    result = formatter.format_single(sample, include_label=True)

    print("\n=== Test: Single sample with False label ===")
    print(result)

    assert "Question: Is the Earth flat?" in result
    assert "Claim: The Earth is flat" in result
    assert "I think this Claim is False" in result


def test_format_icl_single_context():
    """Test ICL formatting with one context example."""
    formatter = PromptFormatter()

    context = TruthfulQASample(
        question="Is water wet?",
        choice="Water is wet",
        predicted_label=True
    )

    query = TruthfulQASample(
        question="Is fire cold?",
        choice="Fire is cold",
        predicted_label=False
    )

    result = formatter.format_icl([context], query)

    print("\n=== Test: ICL with single context example ===")
    print(result)

    # Context should have label
    assert "Question: Is water wet?" in result
    assert "I think this Claim is True" in result

    # Query should not have label
    assert "Question: Is fire cold?" in result
    assert result.count("True") == 1  # Only in context
    assert "False" not in result  # Not in query


def test_format_icl_multiple_context():
    """Test ICL formatting with multiple context examples."""
    formatter = PromptFormatter()

    contexts = [
        TruthfulQASample(question="Q1?", choice="C1", predicted_label=True),
        TruthfulQASample(question="Q2?", choice="C2", predicted_label=False),
        TruthfulQASample(question="Q3?", choice="C3", predicted_label=True)
    ]

    query = TruthfulQASample(question="Q4?", choice="C4", predicted_label=None)

    result = formatter.format_icl(contexts, query)

    print("\n=== Test: ICL with multiple context examples ===")
    print(result)

    # Check all questions present
    assert "Question: Q1?" in result
    assert "Question: Q2?" in result
    assert "Question: Q3?" in result
    assert "Question: Q4?" in result

    # Check double newline separation
    assert "\n\n" in result

    # Check labels only in context (2 True, 1 False)
    assert result.count("True") == 2
    assert result.count("False") == 1


def test_format_icl_empty_context():
    """Test ICL formatting with no context examples."""
    formatter = PromptFormatter()

    query = TruthfulQASample(question="Test?", choice="Test claim", predicted_label=None)

    result = formatter.format_icl([], query)

    assert "Question: Test?" in result
    assert "Claim: Test claim" in result
    assert "True" not in result
    assert "False" not in result


if __name__ == "__main__":
    test_format_single_without_label()
    test_format_single_with_true_label()
    test_format_single_with_false_label()
    test_format_icl_single_context()
    test_format_icl_multiple_context()
    test_format_icl_empty_context()
    print("All tests passed!")
