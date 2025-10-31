"""
API tests for Mutual Predictability Scorer.

These tests make actual API calls to verify the scorer behavior with real model responses.
"""
import sys
from pathlib import Path
import random
import os
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scorer import MutualPredictabilityScorer
from src.data_models import TruthfulQASample
from src.hyperbolic_wrapper import HyperbolicClient
from src.data_loader import load_truthfulqa_data

# Load environment variables for API key
load_dotenv()


def test_api_random_vs_coherent_labels():
    """
    Test that coherent labels get higher mutual predictability than random labels.

    This validates the core assumption: labels that are mutually predictable
    should score higher than random assignments.
    """
    print("\n=== Test: Random vs Coherent Labels (API) ===")

    # Load some samples from train data
    train_data = load_truthfulqa_data('train')

    # Get samples with same question (consistency_id) - these should be coherent
    # Find a question with multiple choices
    consistency_groups = {}
    for sample in train_data:
        cid = sample.consistency_id
        if cid not in consistency_groups:
            consistency_groups[cid] = []
        consistency_groups[cid].append(sample)

    # Find a group with at least 4 samples
    target_group = None
    for cid, samples in consistency_groups.items():
        if len(samples) >= 4:
            target_group = samples[:4]  # Take first 4
            break

    assert target_group is not None, "Need at least 4 samples with same consistency_id"

    print(f"\nUsing question: {target_group[0].question}")
    print("Samples:")
    for i, s in enumerate(target_group):
        print(f"  {i+1}. '{s.choice}' -> {s.predicted_label}")

    # Initialize scorer
    client = HyperbolicClient()
    scorer = MutualPredictabilityScorer(client)

    # Test 1: Coherent labels (original correct labels)
    # Copy samples with predicted_label set from ground_truth_label
    coherent_samples = [
        TruthfulQASample(
            question=s.question,
            choice=s.choice,
            ground_truth_label=s.ground_truth_label,
            predicted_label=s.ground_truth_label,  # Use ground truth as predicted label
            consistency_id=s.consistency_id
        )
        for s in target_group
    ]
    coherent_score = scorer.compute_score(coherent_samples)

    print(f"\nCoherent labels score: {coherent_score:.4f}")

    # Test 2: Random labels (flip labels randomly)
    random_samples = [
        TruthfulQASample(
            question=s.question,
            choice=s.choice,
            ground_truth_label=s.ground_truth_label,
            predicted_label=random.choice([True, False]),  # Random assignment
            consistency_id=s.consistency_id
        )
        for s in target_group
    ]

    print("\nRandomly assigned labels:")
    for i, s in enumerate(random_samples):
        print(f"  {i+1}. '{s.choice}' -> {s.predicted_label}")

    random_score = scorer.compute_score(random_samples)

    print(f"\nRandom labels score: {random_score:.4f}")

    # Coherent labels should generally score higher
    print(f"\nCoherent - Random = {coherent_score - random_score:.4f}")

    # We expect coherent to be higher, but not a strict requirement
    # (depends on the specific examples and randomization)
    print("\n✓ API test completed (scores may vary)")

    return coherent_score, random_score


def test_api_same_question_mutual_predictability():
    """
    Test mutual predictability with samples from the same question.

    Samples from the same question should have higher mutual predictability
    because they share context and the model can infer labels better.
    """
    print("\n=== Test: Same Question Mutual Predictability (API) ===")

    # Load train data
    train_data = load_truthfulqa_data('train')

    # Find samples from the same question
    raw_passport_samples = [
        s for s in train_data
        if "passport" in s.question.lower()
    ][:3]  # Take 3 samples about passports

    assert len(raw_passport_samples) >= 3, "Need at least 3 passport samples"

    # Copy samples with predicted_label set from ground_truth_label
    passport_samples = [
        TruthfulQASample(
            question=s.question,
            choice=s.choice,
            ground_truth_label=s.ground_truth_label,
            predicted_label=s.ground_truth_label,
            consistency_id=s.consistency_id
        )
        for s in raw_passport_samples
    ]

    print(f"\nQuestion: {passport_samples[0].question}")
    print("\nSamples:")
    for i, s in enumerate(passport_samples):
        label_str = "True" if s.predicted_label else "False"
        print(f"  {i+1}. '{s.choice}' -> {label_str}")

    # Initialize scorer
    client = HyperbolicClient()
    scorer = MutualPredictabilityScorer(client)

    # Compute mutual predictability
    score = scorer.compute_score(passport_samples)

    print(f"\nMutual predictability score: {score:.4f}")

    # Check individual logprobs
    print("\nIndividual log probabilities:")
    for i, sample in enumerate(passport_samples):
        other_samples = passport_samples[:i] + passport_samples[i+1:]
        logprob = scorer.get_label_logprob(sample, other_samples)
        label_str = "True" if sample.predicted_label else "False"
        print(f"  Sample {i+1} ({label_str}): {logprob:.4f}")

    # Score should be finite (not -inf)
    assert score != float('-inf'), "Score should be finite with related samples"
    assert score != float('inf'), "Score should not be +inf"

    print("\n✓ API test completed successfully")

    return score


def test_api_different_questions_comparison():
    """
    Test that samples from the same question have better mutual predictability
    than samples from completely different questions.
    """
    print("\n=== Test: Same vs Different Questions (API) ===")

    # Load train data
    train_data = load_truthfulqa_data('train')

    # Get 3 samples from the same question
    consistency_groups = {}
    for sample in train_data:
        cid = sample.consistency_id
        if cid not in consistency_groups:
            consistency_groups[cid] = []
        consistency_groups[cid].append(sample)

    # Find a group with at least 3 samples
    raw_same_question_samples = None
    for cid, samples in consistency_groups.items():
        if len(samples) >= 3:
            raw_same_question_samples = samples[:3]
            break

    assert raw_same_question_samples is not None, "Need samples from same question"

    # Copy with predicted_label set from ground_truth_label
    same_question_samples = [
        TruthfulQASample(
            question=s.question,
            choice=s.choice,
            ground_truth_label=s.ground_truth_label,
            predicted_label=s.ground_truth_label,
            consistency_id=s.consistency_id
        )
        for s in raw_same_question_samples
    ]

    # Get 3 samples from different questions (different consistency_ids)
    different_cids = list(consistency_groups.keys())[:3]
    raw_different_samples = [
        consistency_groups[cid][0] for cid in different_cids
    ]

    different_question_samples = [
        TruthfulQASample(
            question=s.question,
            choice=s.choice,
            ground_truth_label=s.ground_truth_label,
            predicted_label=s.ground_truth_label,
            consistency_id=s.consistency_id
        )
        for s in raw_different_samples
    ]

    print("\n--- Same Question Samples ---")
    print(f"Question: {same_question_samples[0].question}")
    for i, s in enumerate(same_question_samples):
        label_str = "True" if s.predicted_label else "False"
        print(f"  {i+1}. '{s.choice}' -> {label_str}")

    print("\n--- Different Question Samples ---")
    for i, s in enumerate(different_question_samples):
        label_str = "True" if s.predicted_label else "False"
        print(f"  {i+1}. Q: '{s.question[:50]}...'")
        print(f"      C: '{s.choice[:50]}...' -> {label_str}")

    # Initialize scorer
    client = HyperbolicClient()
    scorer = MutualPredictabilityScorer(client)

    # Compute scores
    same_score = scorer.compute_score(same_question_samples)
    different_score = scorer.compute_score(different_question_samples)

    print(f"\nSame question score: {same_score:.4f}")
    print(f"Different questions score: {different_score:.4f}")
    print(f"Difference: {same_score - different_score:.4f}")

    # Same question samples should generally have higher mutual predictability
    # (though not guaranteed, depends on the specific samples)
    print("\n✓ API test completed (comparison informative)")

    return same_score, different_score


def test_api_all_true_vs_mixed_labels():
    """
    Test mutual predictability with all True labels vs mixed True/False.

    All True (or all False) should be more predictable than mixed labels.
    """
    print("\n=== Test: All True vs Mixed Labels (API) ===")

    # Load train data
    train_data = load_truthfulqa_data('train')

    # Get samples that are True (checking ground_truth_label)
    raw_true_samples = [s for s in train_data if s.ground_truth_label][:3]

    # Copy with predicted_label set from ground_truth_label
    true_samples = [
        TruthfulQASample(
            question=s.question,
            choice=s.choice,
            ground_truth_label=s.ground_truth_label,
            predicted_label=True,  # All True
            consistency_id=s.consistency_id
        )
        for s in raw_true_samples
    ]

    # Get mixed samples (some True, some False)
    raw_mixed_samples = []
    true_count = 0
    false_count = 0
    for s in train_data:
        if true_count < 2 and s.ground_truth_label:
            raw_mixed_samples.append((s, True))
            true_count += 1
        elif false_count < 2 and not s.ground_truth_label:
            raw_mixed_samples.append((s, False))
            false_count += 1
        if len(raw_mixed_samples) >= 4:
            break

    mixed_samples = [
        TruthfulQASample(
            question=s.question,
            choice=s.choice,
            ground_truth_label=s.ground_truth_label,
            predicted_label=predicted_label,
            consistency_id=s.consistency_id
        )
        for s, predicted_label in raw_mixed_samples
    ]

    print("\n--- All True Labels ---")
    for i, s in enumerate(true_samples):
        print(f"  {i+1}. Q: '{s.question[:50]}...'")
        print(f"      C: '{s.choice[:50]}...' -> True")

    print("\n--- Mixed Labels ---")
    for i, s in enumerate(mixed_samples):
        label_str = "True" if s.predicted_label else "False"
        print(f"  {i+1}. Q: '{s.question[:50]}...'")
        print(f"      C: '{s.choice[:50]}...' -> {label_str}")

    # Initialize scorer
    client = HyperbolicClient()
    scorer = MutualPredictabilityScorer(client)

    # Compute scores
    all_true_score = scorer.compute_score(true_samples)
    mixed_score = scorer.compute_score(mixed_samples)

    print(f"\nAll True score: {all_true_score:.4f}")
    print(f"Mixed labels score: {mixed_score:.4f}")
    print(f"Difference: {all_true_score - mixed_score:.4f}")

    print("\n✓ API test completed")

    return all_true_score, mixed_score


if __name__ == "__main__":
    print("=" * 60)
    print("Running API tests for Mutual Predictability Scorer")
    print("=" * 60)
    print("\n⚠️  These tests make real API calls and may take some time...")

    # Check if API key is available
    if not os.getenv("HYPERBOLIC_API_KEY"):
        print("\n❌ Error: HYPERBOLIC_API_KEY not found in environment")
        print("Please set it in your .env file")
        sys.exit(1)

    try:
        # Test 1: Random vs Coherent
        coherent_score, random_score = test_api_random_vs_coherent_labels()

        # Test 2: Same question mutual predictability
        same_q_score = test_api_same_question_mutual_predictability()

        # Test 3: Same vs different questions
        same_score, diff_score = test_api_different_questions_comparison()

        # Test 4: All true vs mixed
        all_true_score, mixed_score = test_api_all_true_vs_mixed_labels()

        # Summary
        print("\n" + "=" * 60)
        print("API Test Summary")
        print("=" * 60)
        print(f"Random vs Coherent: {coherent_score:.4f} vs {random_score:.4f}")
        print(f"Same vs Different Q: {same_score:.4f} vs {diff_score:.4f}")
        print(f"All True vs Mixed: {all_true_score:.4f} vs {mixed_score:.4f}")

        print("\n✅ All API tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during API tests: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
