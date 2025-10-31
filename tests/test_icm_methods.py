"""
Simple unit tests for ICM Algorithm methods.
"""
import sys
from pathlib import Path
from unittest.mock import Mock
import math

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.icm_algorithm import ICMAlgorithm
from src.data_models import TruthfulQASample


def create_sample_dataset(n: int = 5):
    """Create a test dataset."""
    return [
        TruthfulQASample(question=f"Q{i}", choice=f"C{i}", ground_truth_label=None, consistency_id=i)
        for i in range(n)
    ]


def test_update_temperature_decreases():
    """Test that temperature decreases over iterations."""
    samples = create_sample_dataset()
    mock_client = Mock()
    icm = ICMAlgorithm(samples, mock_client)

    temp_0 = icm.update_temperature(0)
    temp_10 = icm.update_temperature(10)
    temp_100 = icm.update_temperature(100)

    assert temp_0 > temp_10 > temp_100, \
        f"Temperature should decrease: {temp_0} > {temp_10} > {temp_100}"


def test_update_temperature_respects_minimum():
    """Test that temperature never goes below T_min."""
    samples = create_sample_dataset()
    mock_client = Mock()
    icm = ICMAlgorithm(samples, mock_client, final_temp=0.01)

    # Very large iteration should still respect minimum
    temp = icm.update_temperature(100000)

    assert temp >= 0.01, f"Temperature {temp} should be >= T_min (0.01)"


def test_update_temperature_formula():
    """Test temperature cooling formula from paper."""
    samples = create_sample_dataset()
    mock_client = Mock()
    icm = ICMAlgorithm(
        samples,
        mock_client,
        initial_temp=10.0,
        final_temp=0.01,
        cooling_rate=0.99
    )

    # T = max(T_min, T_0 / (1 + ρ * log(t+1)))
    iteration = 5
    expected = 10.0 / (1 + 0.99 * math.log(iteration + 1))
    expected = max(0.01, expected)

    actual = icm.update_temperature(iteration)

    assert abs(actual - expected) < 1e-9, \
        f"Expected {expected}, got {actual}"


def test_accept_proposal_always_accepts_positive():
    """Test that positive score changes are always accepted."""
    samples = create_sample_dataset()
    mock_client = Mock()
    icm = ICMAlgorithm(samples, mock_client)

    # Positive delta should always be accepted
    assert icm.accept_proposal(delta_score=1.0, temperature=1.0) == True
    assert icm.accept_proposal(delta_score=0.5, temperature=0.1) == True
    assert icm.accept_proposal(delta_score=0.001, temperature=10.0) == True


def test_accept_proposal_negative_probabilistic():
    """Test that negative deltas are accepted probabilistically."""
    samples = create_sample_dataset()
    mock_client = Mock()
    icm = ICMAlgorithm(samples, mock_client)

    # Run many times to check probabilistic behavior
    delta = -1.0
    temp = 1.0

    accepts = 0
    rejects = 0
    for _ in range(100):
        if icm.accept_proposal(delta, temp):
            accepts += 1
        else:
            rejects += 1

    # Should have both accepts and rejects
    assert accepts > 0, "Should accept some negative deltas"
    assert rejects > 0, "Should reject some negative deltas"


def test_accept_proposal_temperature_effect():
    """Test that higher temperature increases acceptance of negative deltas."""
    samples = create_sample_dataset()
    mock_client = Mock()
    icm = ICMAlgorithm(samples, mock_client)

    delta = -2.0

    # Count acceptances at different temperatures
    high_temp_accepts = sum(
        icm.accept_proposal(delta, temperature=10.0)
        for _ in range(100)
    )

    low_temp_accepts = sum(
        icm.accept_proposal(delta, temperature=0.1)
        for _ in range(100)
    )

    # Higher temperature should accept more often
    assert high_temp_accepts > low_temp_accepts, \
        f"High temp ({high_temp_accepts}) should accept more than low temp ({low_temp_accepts})"


def test_create_proposal_adds_new_sample():
    """Test that _create_proposal adds new sample to dataset."""
    samples = create_sample_dataset(3)
    mock_client = Mock()
    icm = ICMAlgorithm(samples, mock_client)

    # Initialize with 2 samples
    icm.labeled_dataset = [
        TruthfulQASample(question=samples[0].question, choice=samples[0].choice,
                        ground_truth_label=samples[0].ground_truth_label,
                        predicted_label=True, consistency_id=samples[0].consistency_id),
        TruthfulQASample(question=samples[1].question, choice=samples[1].choice,
                        ground_truth_label=samples[1].ground_truth_label,
                        predicted_label=False, consistency_id=samples[1].consistency_id)
    ]
    icm.sample_dict = {
        (samples[0].question, samples[0].choice): 0,
        (samples[1].question, samples[1].choice): 1
    }

    # Add a new sample
    new_sample = samples[2]
    proposed, proposed_dict = icm._create_proposal(new_sample, True)

    assert len(proposed) == 3, f"Expected 3 samples, got {len(proposed)}"

    # Check the new sample is included
    questions = [s.question for s in proposed]
    assert "Q2" in questions


def test_create_proposal_updates_existing():
    """Test that _create_proposal updates existing sample's label."""
    samples = create_sample_dataset(3)
    mock_client = Mock()
    icm = ICMAlgorithm(samples, mock_client)

    # Initialize with 2 samples
    icm.labeled_dataset = [
        TruthfulQASample(question=samples[0].question, choice=samples[0].choice,
                        ground_truth_label=samples[0].ground_truth_label,
                        predicted_label=True, consistency_id=samples[0].consistency_id),
        TruthfulQASample(question=samples[1].question, choice=samples[1].choice,
                        ground_truth_label=samples[1].ground_truth_label,
                        predicted_label=False, consistency_id=samples[1].consistency_id)
    ]
    icm.sample_dict = {
        (samples[0].question, samples[0].choice): 0,
        (samples[1].question, samples[1].choice): 1
    }

    # Update existing sample with new label
    proposed, proposed_dict = icm._create_proposal(samples[0], False)

    assert len(proposed) == 2, "Should still have 2 samples"

    # Check label was updated
    sample_0 = [s for s in proposed if s.question == "Q0"][0]
    assert sample_0.predicted_label == False, "Label should be updated to False"


def test_create_proposal_preserves_other_samples():
    """Test that _create_proposal doesn't modify other samples."""
    samples = create_sample_dataset(3)
    mock_client = Mock()
    icm = ICMAlgorithm(samples, mock_client)

    icm.labeled_dataset = [
        TruthfulQASample(question=samples[0].question, choice=samples[0].choice,
                        ground_truth_label=samples[0].ground_truth_label,
                        predicted_label=True, consistency_id=samples[0].consistency_id),
        TruthfulQASample(question=samples[1].question, choice=samples[1].choice,
                        ground_truth_label=samples[1].ground_truth_label,
                        predicted_label=False, consistency_id=samples[1].consistency_id)
    ]
    icm.sample_dict = {
        (samples[0].question, samples[0].choice): 0,
        (samples[1].question, samples[1].choice): 1
    }

    # Update sample 0
    proposed, proposed_dict = icm._create_proposal(samples[0], False)

    # Check sample 1 is unchanged
    sample_1 = [s for s in proposed if s.question == "Q1"][0]
    assert sample_1.predicted_label == False, "Other sample should be unchanged"


def test_sample_example_returns_from_dataset():
    """Test that sample_example returns a sample from the dataset."""
    samples = create_sample_dataset(5)
    mock_client = Mock()
    icm = ICMAlgorithm(samples, mock_client)

    # Sample many times
    sampled_questions = set()
    for _ in range(20):
        sample = icm.sample_example()
        sampled_questions.add(sample.question)

    # All should be from original dataset
    expected_questions = {f"Q{i}" for i in range(5)}
    assert sampled_questions.issubset(expected_questions), \
        "All sampled questions should be from original dataset"


def test_sample_example_has_variety():
    """Test that sample_example produces different samples over time."""
    samples = create_sample_dataset(10)
    mock_client = Mock()
    icm = ICMAlgorithm(samples, mock_client)

    sampled_questions = [icm.sample_example().question for _ in range(30)]
    unique_samples = set(sampled_questions)

    # Should get more than 1 unique sample
    assert len(unique_samples) > 1, \
        f"Should sample different examples, got {len(unique_samples)} unique"


def test_accept_proposal_formula():
    """Test that acceptance probability follows exp(Δ/T) formula."""
    samples = create_sample_dataset()
    mock_client = Mock()
    icm = ICMAlgorithm(samples, mock_client)

    delta = -1.0
    temp = 2.0

    # Expected acceptance probability: exp(-1.0 / 2.0) = exp(-0.5) ≈ 0.606
    expected_prob = math.exp(delta / temp)

    # Run many trials
    accepts = sum(icm.accept_proposal(delta, temp) for _ in range(1000))
    actual_prob = accepts / 1000

    # Should be close to expected (within 10%)
    assert abs(actual_prob - expected_prob) < 0.1, \
        f"Acceptance rate {actual_prob} should be close to {expected_prob}"


if __name__ == "__main__":
    print("Running ICM methods tests...\n")

    test_update_temperature_decreases()
    print("✓ Test: update_temperature_decreases")

    test_update_temperature_respects_minimum()
    print("✓ Test: update_temperature_respects_minimum")

    test_update_temperature_formula()
    print("✓ Test: update_temperature_formula")

    test_accept_proposal_always_accepts_positive()
    print("✓ Test: accept_proposal_always_accepts_positive")

    test_accept_proposal_negative_probabilistic()
    print("✓ Test: accept_proposal_negative_probabilistic")

    test_accept_proposal_temperature_effect()
    print("✓ Test: accept_proposal_temperature_effect")

    test_create_proposal_adds_new_sample()
    print("✓ Test: create_proposal_adds_new_sample")

    test_create_proposal_updates_existing()
    print("✓ Test: create_proposal_updates_existing")

    test_create_proposal_preserves_other_samples()
    print("✓ Test: create_proposal_preserves_other_samples")

    test_sample_example_returns_from_dataset()
    print("✓ Test: sample_example_returns_from_dataset")

    test_sample_example_has_variety()
    print("✓ Test: sample_example_has_variety")

    test_accept_proposal_formula()
    print("✓ Test: accept_proposal_formula")

    print("\n✅ All tests passed!")
