"""
Tests for ICM Algorithm Random Initialization.

Tests the initialize_random_labels method from Algorithm 1 in the paper.
"""
import sys
from pathlib import Path
from unittest.mock import Mock
import random

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.icm_algorithm import ICMAlgorithm
from src.data_models import TruthfulQASample


def create_sample_dataset(n: int = 10):
    """Create a test dataset with n samples."""
    return [
        TruthfulQASample(
            question=f"Question {i}?",
            choice=f"Choice {i}",
            ground_truth_label=None,
            consistency_id=i
        )
        for i in range(n)
    ]


def test_initialize_returns_correct_count():
    """Test that initialize_random_labels returns exactly k samples."""
    samples = create_sample_dataset(10)
    mock_client = Mock()
    icm = ICMAlgorithm(samples, mock_client)

    # Test default k=8
    labeled = icm.initialize_random_labels(k=8)
    assert len(labeled) == 8, f"Expected 8 samples, got {len(labeled)}"

    # Test different values of k
    for k in [1, 3, 5, 10]:
        labeled = icm.initialize_random_labels(k=k)
        assert len(labeled) == k, f"Expected {k} samples, got {len(labeled)}"


def test_initialize_returns_icm_labeled_samples():
    """Test that returned samples are TruthfulQASample objects with predicted labels."""
    samples = create_sample_dataset(10)
    mock_client = Mock()
    icm = ICMAlgorithm(samples, mock_client)

    labeled = icm.initialize_random_labels(k=5)

    for sample in labeled:
        assert isinstance(sample, TruthfulQASample), \
            f"Expected TruthfulQASample, got {type(sample)}"
        assert isinstance(sample.predicted_label, bool), \
            f"predicted_label should be bool, got {type(sample.predicted_label)}"


def test_initialize_samples_from_input():
    """Test that all returned samples come from input dataset."""
    samples = create_sample_dataset(10)
    mock_client = Mock()
    icm = ICMAlgorithm(samples, mock_client)

    labeled = icm.initialize_random_labels(k=5)

    # Check each labeled sample is from original dataset
    original_questions = {s.question for s in samples}
    for labeled_sample in labeled:
        assert labeled_sample.question in original_questions, \
            "Labeled sample not from original dataset"


def test_initialize_no_duplicates():
    """Test that initialize_random_labels doesn't return duplicate samples."""
    samples = create_sample_dataset(10)
    mock_client = Mock()
    icm = ICMAlgorithm(samples, mock_client)

    labeled = icm.initialize_random_labels(k=8)

    # Check for duplicates using question+choice as key
    seen = set()
    for sample in labeled:
        key = (sample.question, sample.choice)
        assert key not in seen, f"Duplicate sample found: {key}"
        seen.add(key)


def test_initialize_labels_are_boolean():
    """Test that predicted labels are True or False."""
    samples = create_sample_dataset(10)
    mock_client = Mock()
    icm = ICMAlgorithm(samples, mock_client)

    labeled = icm.initialize_random_labels(k=5)

    for sample in labeled:
        assert sample.predicted_label in [True, False], \
            f"Label should be True or False, got {sample.predicted_label}"


def test_initialize_has_randomness():
    """Test that initialization produces different results across runs."""
    samples = create_sample_dataset(20)
    mock_client = Mock()
    icm = ICMAlgorithm(samples, mock_client)

    # Run initialization multiple times
    runs = []
    for _ in range(5):
        labeled = icm.initialize_random_labels(k=10)
        # Create signature: (questions selected, labels assigned)
        questions = tuple(s.question for s in labeled)
        labels = tuple(s.predicted_label for s in labeled)
        runs.append((questions, labels))

    # Check that at least some runs differ
    unique_runs = len(set(runs))
    assert unique_runs > 1, \
        f"Expected some variation across runs, got {unique_runs} unique out of 5"


def test_initialize_respects_default_k():
    """Test that default k=8 is used when not specified."""
    samples = create_sample_dataset(20)
    mock_client = Mock()
    icm = ICMAlgorithm(samples, mock_client)

    # From paper Appendix A.1: default initialization is k=8
    labeled = icm.initialize_random_labels()  # Use default
    assert len(labeled) == 8, f"Default k should be 8, got {len(labeled)}"


def test_initialize_k_equals_dataset_size():
    """Test edge case where k equals total dataset size."""
    samples = create_sample_dataset(5)
    mock_client = Mock()
    icm = ICMAlgorithm(samples, mock_client)

    labeled = icm.initialize_random_labels(k=5)

    assert len(labeled) == 5, "Should return all samples when k=n"

    # All original samples should be included
    original_questions = {s.question for s in samples}
    labeled_questions = {s.question for s in labeled}
    assert original_questions == labeled_questions, \
        "Should include all original samples"


def test_initialize_k_equals_one():
    """Test edge case where k=1."""
    samples = create_sample_dataset(10)
    mock_client = Mock()
    icm = ICMAlgorithm(samples, mock_client)

    labeled = icm.initialize_random_labels(k=1)

    assert len(labeled) == 1, "Should return 1 sample when k=1"
    assert isinstance(labeled[0], TruthfulQASample), \
        "Should return TruthfulQASample"


def test_initialize_preserves_original_sample_data():
    """Test that original sample data is preserved in TruthfulQASample."""
    samples = [
        TruthfulQASample(question="Q1", choice="C1", ground_truth_label=None, consistency_id=100),
        TruthfulQASample(question="Q2", choice="C2", ground_truth_label=None, consistency_id=200),
        TruthfulQASample(question="Q3", choice="C3", ground_truth_label=None, consistency_id=300),
    ]
    mock_client = Mock()
    icm = ICMAlgorithm(samples, mock_client)

    labeled = icm.initialize_random_labels(k=3)

    # Check all original data is preserved
    for labeled_sample in labeled:
        assert labeled_sample.question in ["Q1", "Q2", "Q3"]
        assert labeled_sample.choice in ["C1", "C2", "C3"]
        assert labeled_sample.ground_truth_label is None  # Original label should be None (unlabeled)
        assert labeled_sample.consistency_id in [100, 200, 300]


def test_initialize_label_distribution():
    """Test that labels have reasonable distribution (not all same)."""
    samples = create_sample_dataset(50)
    mock_client = Mock()
    icm = ICMAlgorithm(samples, mock_client)

    # Run multiple initializations and collect labels
    all_labels = []
    for _ in range(10):
        labeled = icm.initialize_random_labels(k=20)
        all_labels.extend([s.predicted_label for s in labeled])

    # Count True and False
    true_count = sum(all_labels)
    false_count = len(all_labels) - true_count

    # Both should be present (extremely unlikely to get all same by chance)
    assert true_count > 0, "Should have some True labels"
    assert false_count > 0, "Should have some False labels"

    # Rough balance check (not too skewed)
    ratio = true_count / len(all_labels)
    assert 0.3 < ratio < 0.7, \
        f"Labels should be roughly balanced, got {true_count}/{len(all_labels)} True"


def test_initialize_uses_random_choice():
    """Test that both True and False labels can appear in single run."""
    samples = create_sample_dataset(20)
    mock_client = Mock()
    icm = ICMAlgorithm(samples, mock_client)

    # With k=20, extremely unlikely to get all same label
    labeled = icm.initialize_random_labels(k=20)
    labels = [s.predicted_label for s in labeled]

    has_true = True in labels
    has_false = False in labels

    # Both should appear (probability of all same is 2 * (0.5^20) ≈ 0)
    assert has_true and has_false, \
        f"Expected both True and False labels, got {sum(labels)} True out of {len(labels)}"


def test_initialize_with_specific_samples():
    """Test initialization with specific known samples."""
    samples = [
        TruthfulQASample(question="Is sky blue?", choice="Yes", ground_truth_label=None, consistency_id=1),
        TruthfulQASample(question="Is water wet?", choice="Yes", ground_truth_label=None, consistency_id=1),
        TruthfulQASample(question="Is earth flat?", choice="No", ground_truth_label=None, consistency_id=2),
    ]
    mock_client = Mock()
    icm = ICMAlgorithm(samples, mock_client)

    labeled = icm.initialize_random_labels(k=3)

    # Check structure
    assert len(labeled) == 3
    questions = [s.question for s in labeled]
    assert "Is sky blue?" in questions
    assert "Is water wet?" in questions
    assert "Is earth flat?" in questions

    # Check labels are assigned
    for sample in labeled:
        assert sample.predicted_label is not None
        assert isinstance(sample.predicted_label, bool)


def test_initialize_paper_hyperparameter():
    """Test that default k=8 matches paper's hyperparameter (Sec 2.3)."""
    samples = create_sample_dataset(20)
    mock_client = Mock()
    icm = ICMAlgorithm(samples, mock_client)

    # From paper: "we find that a small number (e.g., K = 8), often strikes a good balance"
    labeled = icm.initialize_random_labels()

    assert len(labeled) == 8, \
        "Default k should be 8 as specified in paper"


if __name__ == "__main__":
    print("Running ICM random initialization tests...\n")

    test_initialize_returns_correct_count()
    print("✓ Test: initialize_returns_correct_count")

    test_initialize_returns_icm_labeled_samples()
    print("✓ Test: initialize_returns_icm_labeled_samples")

    test_initialize_samples_from_input()
    print("✓ Test: initialize_samples_from_input")

    test_initialize_no_duplicates()
    print("✓ Test: initialize_no_duplicates")

    test_initialize_labels_are_boolean()
    print("✓ Test: initialize_labels_are_boolean")

    test_initialize_has_randomness()
    print("✓ Test: initialize_has_randomness")

    test_initialize_respects_default_k()
    print("✓ Test: initialize_respects_default_k")

    test_initialize_k_equals_dataset_size()
    print("✓ Test: initialize_k_equals_dataset_size")

    test_initialize_k_equals_one()
    print("✓ Test: initialize_k_equals_one")

    test_initialize_preserves_original_sample_data()
    print("✓ Test: initialize_preserves_original_sample_data")

    test_initialize_label_distribution()
    print("✓ Test: initialize_label_distribution")

    test_initialize_uses_random_choice()
    print("✓ Test: initialize_uses_random_choice")

    test_initialize_with_specific_samples()
    print("✓ Test: initialize_with_specific_samples")

    test_initialize_paper_hyperparameter()
    print("✓ Test: initialize_paper_hyperparameter")

    print("\n✅ All tests passed!")
