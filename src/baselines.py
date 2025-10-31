"""
Baseline methods for comparison with ICM.

Implements zero-shot baselines on both base and instruct models.
"""
import random
from typing import List
from .data_models import TruthfulQASample
from .hyperbolic_wrapper import HyperbolicClient
from .data_loader import load_truthfulqa_data
from .prompt_formatter import PromptFormatter


def zero_shot(samples: List[TruthfulQASample], model_type: str) -> List[TruthfulQASample]:
    """Run zero-shot baseline - direct prediction without any context."""
    results = []
    total = len(samples)
    client = HyperbolicClient(model_type=model_type)
    formatter = PromptFormatter()

    for i, sample in enumerate(samples):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{total} samples")

        # Format prompt for zero-shot
        prompt = formatter.format_single(sample, include_label=False)

        # Generate response
        response = client.generate(prompt, max_tokens=1, temperature=0.0, logprobs=0)
        text = client.get_text(response).strip()

        # Parse response to get prediction
        predicted_label = text.lower().startswith('true')

        results.append(TruthfulQASample(
            question=sample.question,
            choice=sample.choice,
            ground_truth_label=sample.ground_truth_label,
            predicted_label=predicted_label,
            consistency_id=sample.consistency_id
        ))

    return results


def golden_labels(
    samples: List[TruthfulQASample],
    model_type: str,
    num_context_samples: int,
    split: str = "test",
    seed: int = 42
) -> List[TruthfulQASample]:
    """
    Golden labels baseline: randomly sample a fixed set of samples with ground truth labels
    and use them as context for all predictions.

    Args:
        samples: All samples to evaluate
        model_type: Model type to use ('base' or 'instruct')
        num_context_samples: Number of samples to use as golden context
        split: Split being evaluated ('train' or 'test')
        seed: Random seed for reproducible sampling

    Returns:
        List of samples with predicted labels
    """
    # Determine context pool based on split
    if split == "test":
        # For test, use all train samples as context pool
        context_pool = load_truthfulqa_data(split="train", max_samples=None)
    else:
        # For train, use train samples excluding those being evaluated
        all_train = load_truthfulqa_data(split="train", max_samples=None)
        # Create set of sample keys being evaluated
        eval_keys = {(s.question, s.choice) for s in samples}
        # Filter out samples being evaluated
        context_pool = [s for s in all_train if (s.question, s.choice) not in eval_keys]

    if not all(sample.ground_truth_label is not None for sample in context_pool):
        raise ValueError("All context samples must have ground_truth_label for golden labels baseline")

    # Randomly sample context samples (with seed for reproducibility)
    random.seed(seed)
    selected_context = random.sample(context_pool, min(num_context_samples, len(context_pool)))

    # Set predicted_label to ground_truth_label for context samples
    context_with_labels = [
        TruthfulQASample(
            question=sample.question,
            choice=sample.choice,
            ground_truth_label=sample.ground_truth_label,
            predicted_label=sample.ground_truth_label,  # Use golden label
            consistency_id=sample.consistency_id
        )
        for sample in selected_context
    ]

    results = []
    total = len(samples)
    client = HyperbolicClient(model_type=model_type)
    formatter = PromptFormatter()

    for i, sample in enumerate(samples):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{total} samples")

        # Format prompt with golden context
        prompt = formatter.format_icl(context_with_labels, sample)

        # Generate response
        response = client.generate(prompt, max_tokens=1, temperature=0.0, logprobs=0)
        text = client.get_text(response).strip()

        # Parse response to get prediction
        predicted_label = text.lower().startswith('true')

        results.append(TruthfulQASample(
            question=sample.question,
            choice=sample.choice,
            ground_truth_label=sample.ground_truth_label,
            predicted_label=predicted_label,
            consistency_id=sample.consistency_id
        ))

    return results

