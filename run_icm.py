"""
Run ICM algorithm on TruthfulQA data.
"""
import os
from dotenv import load_dotenv
from src.data_loader import load_truthfulqa_data
from src.hyperbolic_wrapper import HyperbolicClient
from src.icm_algorithm import ICMAlgorithm

# Load environment variables
load_dotenv()


def run_icm_experiment(
    max_samples: int = 200,
    num_iterations: int = 800,
    initial_k: int = 8,
    initial_temp: float = 10.0,
    final_temp: float = 0.01,
    cooling_rate: float = 0.99,
    split: str = "train"
):
    """
    Run ICM algorithm on TruthfulQA data.

    Args:
        max_samples: Number of samples to load from dataset
        num_iterations: Number of iterations for simulated annealing
        initial_k: Number of initial random labels
        initial_temp: Initial temperature for simulated annealing
        final_temp: Final temperature for simulated annealing
        cooling_rate: Cooling rate for temperature schedule
        split: Dataset split to use ('train' or 'test')
    """
    client = HyperbolicClient()

    # Load data
    print(f"Loading {max_samples} samples from {split} split...")
    samples = load_truthfulqa_data(split=split, max_samples=max_samples)
    print(f"Loaded {len(samples)} samples")

    # Initialize ICM algorithm
    log_file = f"icm_log_{split}_{max_samples}samples_{num_iterations}iters.json"
    icm = ICMAlgorithm(
        unlabeled_samples=samples,
        model_client=client,
        initial_temp=initial_temp,
        final_temp=final_temp,
        cooling_rate=cooling_rate,
        log_file=log_file
    )

    # Run algorithm
    print(f"\nRunning ICM for {num_iterations} iterations...")
    labeled_samples = icm.run(num_iterations=num_iterations, initial_k=initial_k)

    # Calculate accuracy
    correct = sum(1 for s in labeled_samples if s.predicted_label == s.ground_truth_label)
    accuracy = correct / len(labeled_samples) if labeled_samples else 0.0

    # Print results
    print(f"\n--- Results ---")
    print(f"Final dataset size: {len(labeled_samples)}")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{len(labeled_samples)})")
    print(f"\nFirst 5 labeled samples:")
    for i, sample in enumerate(labeled_samples[:5]):
        gt_str = "✓" if sample.predicted_label == sample.ground_truth_label else "✗"
        print(f"{i+1}. Q: {sample.question[:60]}...")
        print(f"   Claim: {sample.choice[:60]}...")
        print(f"   Predicted: {sample.predicted_label}, Ground Truth: {sample.ground_truth_label} {gt_str}")
        print()

    return labeled_samples


if __name__ == "__main__":
    # Small test run
    labeled_samples = run_icm_experiment(
        max_samples=50,
        num_iterations=200,
        initial_k=8
    )
