"""
Compare ICM algorithm with baseline methods on TruthfulQA.
"""
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from src.data_loader import load_truthfulqa_data
from src.hyperbolic_wrapper import HyperbolicClient
from src.icm_algorithm import ICMAlgorithm
from src import baselines

load_dotenv()

# ICM Configuration
NUM_ITERATIONS = 200
INITIAL_K = 8
INITIAL_TEMP = 10.0
FINAL_TEMP = 0.01
COOLING_RATE = 0.99


def calculate_accuracy(samples):
    """Calculate accuracy by comparing predicted and ground truth labels."""
    if not samples:
        return 0.0
    correct = sum(1 for s in samples if s.predicted_label == s.ground_truth_label)
    return correct / len(samples)


def run_comparison(
    max_samples: int = 100,
    num_golden_samples: int = 8,
    split: str = "test",
    seed: int = 42,
    resume_from: str = None
):
    """Run comparison between ICM and baselines.

    Args:
        resume_from: Path to checkpoint file to resume ICM from (e.g., "checkpoints/checkpoint_iter_100.json")
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load data
    print(f"\n{'='*60}")
    print(f"Loading {max_samples} samples from {split} split...")
    print(f"{'='*60}\n")
    samples = load_truthfulqa_data(split=split, max_samples=max_samples)
    print(f"Loaded {len(samples)} samples\n")

    # Try to load existing baseline results if resuming
    zero_shot_base_accuracy = None
    zero_shot_instruct_accuracy = None
    golden_accuracy = None
    
    if resume_from:
        # Look for existing baseline results
        baseline_files = [f for f in os.listdir("results") if f.startswith("baseline_results_") and split in f]
        if baseline_files:
            baseline_file = os.path.join("results", sorted(baseline_files)[-1])  # Get most recent
            try:
                with open(baseline_file, 'r') as f:
                    baseline_data = json.load(f)
                    zero_shot_base_accuracy = baseline_data['results']['zero_shot_base']
                    zero_shot_instruct_accuracy = baseline_data['results']['zero_shot_instruct']
                    golden_accuracy = baseline_data['results']['golden_labels_base']
                    print(f"Loaded baseline results from {baseline_file}\n")
            except Exception as e:
                print(f"Warning: Could not load baseline results: {e}")
                print("Will only report ICM results.\n")
    
    if not resume_from:
        # Zero-shot BASE
        print(f"\n{'='*60}")
        print("Running Zero-Shot Baseline (BASE)...")
        print(f"{'='*60}\n")
        zero_shot_base_results = baselines.zero_shot(samples, model_type="base")
        zero_shot_base_accuracy = calculate_accuracy(zero_shot_base_results)
        print(f"\nZero-shot (BASE) Accuracy: {zero_shot_base_accuracy:.2%}")

        # Zero-shot INSTRUCT
        print(f"\n{'='*60}")
        print("Running Zero-Shot Baseline (INSTRUCT)...")
        print(f"{'='*60}\n")
        zero_shot_instruct_results = baselines.zero_shot(samples, model_type="instruct")
        zero_shot_instruct_accuracy = calculate_accuracy(zero_shot_instruct_results)
        print(f"\nZero-shot (INSTRUCT) Accuracy: {zero_shot_instruct_accuracy:.2%}")

        # Golden labels BASE
        print(f"\n{'='*60}")
        print(f"Running Golden Labels Baseline (BASE, k={num_golden_samples})...")
        print(f"{'='*60}\n")
        golden_results = baselines.golden_labels(
            samples,
            model_type="base",
            num_context_samples=num_golden_samples,
            split=split,
            seed=seed
        )
        golden_accuracy = calculate_accuracy(golden_results)
        print(f"\nGolden Labels (BASE) Accuracy: {golden_accuracy:.2%}")

        baseline_results = {
            "config": {
                "max_samples": max_samples,
                "num_golden_samples": num_golden_samples,
                "split": split,
                "seed": seed,
                "timestamp": timestamp
            },
            "results": {
                "zero_shot_base": zero_shot_base_accuracy,
                "zero_shot_instruct": zero_shot_instruct_accuracy,
                "golden_labels_base": golden_accuracy
            }
        }
        os.makedirs("results", exist_ok=True)
        baseline_file = f"results/baseline_results_{split}_{max_samples}samples_{timestamp}.json"
        with open(baseline_file, 'w') as f:
            json.dump(baseline_results, f, indent=2)
        print(f"\nBaseline results saved to {baseline_file}")

    # ICM BASE
    print(f"\n{'='*60}")
    print(f"Running ICM Algorithm (BASE)...")
    print(f"{'='*60}\n")
    os.makedirs("results", exist_ok=True)
    icm_log_file = f"results/icm_comparison_{split}_{max_samples}samples_{timestamp}.json"
    client_base = HyperbolicClient(model_type="base")
    icm = ICMAlgorithm(
        unlabeled_samples=samples,
        model_client=client_base,
        initial_temp=INITIAL_TEMP,
        final_temp=FINAL_TEMP,
        cooling_rate=COOLING_RATE,
        log_file=icm_log_file
    )
    labeled_dataset = icm.run(num_iterations=NUM_ITERATIONS, initial_k=INITIAL_K, resume_from=resume_from)
    icm_accuracy = calculate_accuracy(labeled_dataset)
    print(f"\nICM (BASE) Accuracy: {icm_accuracy:.2%}")
    print(f"ICM Final Dataset Size: {len(labeled_dataset)}")

    # Summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}\n")
    print(f"{'Method':<30} {'Accuracy':<12}")
    print(f"{'-'*42}")
    
    if zero_shot_base_accuracy is not None:
        print(f"{'Zero-shot (BASE)':<30} {zero_shot_base_accuracy:.2%}")
        print(f"{'Zero-shot (INSTRUCT)':<30} {zero_shot_instruct_accuracy:.2%}")
        print(f"{'Golden Labels (BASE, k='}{num_golden_samples}{')':<19} {golden_accuracy:.2%}")
    print(f"{'ICM (BASE)':<30} {icm_accuracy:.2%}")
    print()

    # Save results
    if zero_shot_base_accuracy is not None:
        # Save full comparison results
        results = {
            "config": {
                "max_samples": max_samples,
                "num_golden_samples": num_golden_samples,
                "split": split,
                "seed": seed,
                "timestamp": timestamp,
                "icm": {
                    "num_iterations": NUM_ITERATIONS,
                    "initial_k": INITIAL_K,
                    "initial_temp": INITIAL_TEMP,
                    "final_temp": FINAL_TEMP,
                    "cooling_rate": COOLING_RATE
                }
            },
            "results": {
                "zero_shot_base": zero_shot_base_accuracy,
                "zero_shot_instruct": zero_shot_instruct_accuracy,
                "golden_labels_base": golden_accuracy,
                "icm_base": icm_accuracy
            }
        }
        results_file = f"results/comparison_results_{split}_{max_samples}samples_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_file}")
    else:
        # Only ICM results available (resumed from checkpoint)
        results = {
            "config": {
                "max_samples": max_samples,
                "split": split,
                "timestamp": timestamp,
                "icm": {
                    "num_iterations": NUM_ITERATIONS,
                    "initial_k": INITIAL_K,
                    "initial_temp": INITIAL_TEMP,
                    "final_temp": FINAL_TEMP,
                    "cooling_rate": COOLING_RATE
                }
            },
            "results": {
                "icm_base": icm_accuracy
            }
        }
        results_file = f"results/icm_only_results_{split}_{max_samples}samples_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"ICM results saved to {results_file}")
        print("Note: Run without resume_from to generate full comparison with baselines.")

    return results


if __name__ == "__main__":
    results = run_comparison()
