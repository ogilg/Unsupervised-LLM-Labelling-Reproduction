"""
Internal Coherence Maximization (ICM) Algorithm.

Implements Algorithm 1 from "Unsupervised Elicitation of Language Models"
"""
import random
import math
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from .data_models import TruthfulQASample
from .scorer import MutualPredictabilityScorer


class ICMAlgorithm:
    """Simulated annealing-based search for optimal label assignment."""

    def __init__(
        self,
        unlabeled_samples: List[TruthfulQASample],
        model_client,
        initial_temp: float = 10.0,
        final_temp: float = 0.01,
        cooling_rate: float = 0.99,
        log_file: Optional[str] = None,
        checkpoint_dir: str = "checkpoints"
    ):
        self.unlabeled_samples = unlabeled_samples
        self.model_client = model_client
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.scorer = MutualPredictabilityScorer(model_client)
        self.log_file = log_file
        self.checkpoint_dir = checkpoint_dir

        self.labeled_dataset = []
        self.sample_dict = {}

    def initialize_random_labels(self, k: int = 8) -> List[TruthfulQASample]:
        """
        Randomly select and label K examples.

        Args:
            k: Number of examples to initialize with

        Returns:
            List of K randomly labeled samples
        """
        selected = random.sample(self.unlabeled_samples, k)

        self.sample_dict = {}  # Clear dict for fresh initialization
        labeled = []
        for i, sample in enumerate(selected):
            labeled_sample = TruthfulQASample(
                question=sample.question,
                choice=sample.choice,
                ground_truth_label=sample.ground_truth_label,
                predicted_label=random.choice([True, False]),
                consistency_id=sample.consistency_id
            )
            labeled.append(labeled_sample)
            self.sample_dict[(sample.question, sample.choice)] = i

        return labeled

    def update_temperature(self, iteration: int) -> float:
        """
        Update temperature using cooling schedule.

        T = max(T_min, T_0 / (1 + ρ * log(t)))

        Args:
            iteration: Current iteration number

        Returns:
            Current temperature
        """
        temp = self.initial_temp / (1 + self.cooling_rate * math.log(iteration + 1))
        return max(self.final_temp, temp)

    def sample_example(self) -> TruthfulQASample:
        """
        Sample an example to label.

        Returns:
            Sampled example (can be unlabeled or previously labeled)
        """
        return random.choice(self.unlabeled_samples)

    def assign_label(self, sample: TruthfulQASample, context: List[TruthfulQASample]) -> bool:
        """
        Assign label to sample using model.

        y_hat = argmax P(y | x, context)

        Args:
            sample: Sample to label
            context: Other labeled samples for context

        Returns:
            Predicted label (True/False)
        """
        true_logprob, false_logprob = self.scorer.get_both_logprobs(sample, context)

        return true_logprob > false_logprob

    def accept_proposal(self, delta_score: float, temperature: float) -> bool:
        """
        Decide whether to accept a new labeling.
        """
        if delta_score > 0:
            return True

        acceptance_prob = math.exp(delta_score / temperature)
        return random.random() < acceptance_prob

    def save_checkpoint(self, iteration: int, current_score: float, total_acceptances: int):
        """Save checkpoint to disk."""
        Path(self.checkpoint_dir).mkdir(exist_ok=True)
        checkpoint = {
            "iteration": iteration,
            "labeled_dataset": [
                {
                    "question": s.question,
                    "choice": s.choice,
                    "ground_truth_label": s.ground_truth_label,
                    "predicted_label": s.predicted_label,
                    "consistency_id": s.consistency_id
                }
                for s in self.labeled_dataset
            ],
            "sample_dict": {f"{k[0]}|||{k[1]}": v for k, v in self.sample_dict.items()},
            "current_score": current_score,
            "total_acceptances": total_acceptances
        }
        checkpoint_file = os.path.join(self.checkpoint_dir, f"checkpoint_iter_{iteration}.json")
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def load_checkpoint(self, checkpoint_file: str) -> Tuple[int, float, int]:
        """Load checkpoint from disk. Returns (start_iteration, current_score, total_acceptances)."""
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)

        self.labeled_dataset = [
            TruthfulQASample(
                question=s["question"],
                choice=s["choice"],
                ground_truth_label=s["ground_truth_label"],
                predicted_label=s["predicted_label"],
                consistency_id=s["consistency_id"]
            )
            for s in checkpoint["labeled_dataset"]
        ]
        self.sample_dict = {
            tuple(k.split("|||")): v for k, v in checkpoint["sample_dict"].items()
        }
        return checkpoint["iteration"], checkpoint["current_score"], checkpoint["total_acceptances"]

    def run(self, num_iterations: int, initial_k: int = 8, resume_from: Optional[str] = None) -> List[TruthfulQASample]:
        """
        Run the ICM algorithm.

        Args:
            num_iterations: Number of iterations to run
            initial_k: Number of examples to initialize with
            resume_from: Path to checkpoint file to resume from

        Returns:
            Final labeled dataset
        """
        import time
        start_time = time.time()

        log_data = {
            "config": {
                "num_iterations": num_iterations,
                "initial_k": initial_k,
                "initial_temp": self.initial_temp,
                "final_temp": self.final_temp,
                "cooling_rate": self.cooling_rate,
                "num_unlabeled_samples": len(self.unlabeled_samples)
            },
            "iterations": []
        }

        # Resume from checkpoint or initialize
        if resume_from:
            print(f"Resuming from checkpoint: {resume_from}")
            start_iteration, current_score, total_acceptances = self.load_checkpoint(resume_from)
            start_iteration += 1  # Start from next iteration
            initial_score = current_score
            print(f"Resumed at iteration {start_iteration}")
        else:
            # Step 1: Initialize with K randomly labeled examples
            self.labeled_dataset = self.initialize_random_labels(initial_k)
            print(f"Initialized with {initial_k} random labels")

            initial_labels = [
                {"question": s.question, "choice": s.choice, "label": s.predicted_label}
                for s in self.labeled_dataset
            ]
            log_data["initialization"] = initial_labels

            # Initialize monitoring variables
            total_acceptances = 0
            start_iteration = 0

            # Compute initial score once
            current_score = self.scorer.compute_score(self.labeled_dataset)
            initial_score = current_score

        recent_acceptances = []  # Track last 10 for recent acceptance rate

        print("\n" + "="*90)
        print(f"{'Iter':>6} | {'Temp':>8} | {'Energy':>10} | {'Delta':>8} | {'Accept':>6} | {'Acc Rate':>8} | {'Size':>5} | {'Time':>7} | {'Cache':>7}")
        print("="*90)

        # Step 2: Main simulated annealing loop
        checkpoint_start = time.time()
        for iteration in range(start_iteration, num_iterations):
            iter_start = time.time()

            # Update temperature
            temperature = self.update_temperature(iteration)

            # Sample an example to label
            target_sample = self.sample_example()

            # Assign label using model
            new_label = self.assign_label(target_sample, self.labeled_dataset)

            # Create proposed dataset
            proposed_dataset, proposed_dict = self._create_proposal(target_sample, new_label)

            # Calculate proposed score only
            proposed_score = self.scorer.compute_score(proposed_dataset)
            delta = proposed_score - current_score

            # Check if this is an update or new addition
            sample_key = (target_sample.question, target_sample.choice)
            is_update = sample_key in self.sample_dict

            # Accept or reject proposal
            accepted = self.accept_proposal(delta, temperature)
            if accepted:
                self.labeled_dataset = proposed_dataset
                self.sample_dict = proposed_dict
                current_score = proposed_score  # Update cached score
                self.scorer.clear_cache()  # Clear cache since dataset changed
                total_acceptances += 1

            # Update recent acceptances (rolling window of 10)
            recent_acceptances.append(1 if accepted else 0)
            if len(recent_acceptances) > 10:
                recent_acceptances.pop(0)

            # Calculate acceptance rates
            cumulative_rate = total_acceptances / (iteration + 1)
            recent_rate = sum(recent_acceptances) / len(recent_acceptances)

            iter_time = time.time() - iter_start
            cache_stats = self.scorer.get_cache_stats()

            # Log this iteration
            iter_log = {
                "iteration": iteration,
                "temperature": temperature,
                "current_score": current_score,
                "proposed_score": proposed_score,
                "accepted": accepted,
                "dataset_size": len(self.labeled_dataset),
                "cumulative_acceptance_rate": cumulative_rate,
                "recent_acceptance_rate": recent_rate,
                "iter_time": iter_time,
                "cache_hits": cache_stats["hits"],
                "cache_hit_rate": cache_stats["hit_rate"],
                "target_sample": {
                    "question": target_sample.question,
                    "choice": target_sample.choice,
                    "proposed_label": new_label
                },
                "is_update": is_update
            }
            log_data["iterations"].append(iter_log)

            # Save checkpoint every 20 iterations
            if iteration > 0 and iteration % 20 == 0:
                self.save_checkpoint(iteration, current_score, total_acceptances)
                print(f"Checkpoint saved at iteration {iteration}")

            # Log progress every 10 iterations
            if iteration % 10 == 0 or iteration < 5:
                accept_symbol = "✓" if accepted else "✗"
                checkpoint_time = time.time() - checkpoint_start
                num_iters_since_checkpoint = 10 if iteration >= 10 else iteration + 1
                avg_time = checkpoint_time / num_iters_since_checkpoint
                print(f"{iteration:6d} | {temperature:8.3f} | {current_score:10.2f} | {delta:8.2f} | "
                      f"{accept_symbol:>6} | {cumulative_rate:7.1%} | {len(self.labeled_dataset):5d} | "
                      f"{avg_time:5.2f}s | {cache_stats['hit_rate']:.1%}")
                checkpoint_start = time.time()

        final_score = self.scorer.compute_score(self.labeled_dataset)
        energy_change = final_score - initial_score if initial_score is not None else 0
        total_time = time.time() - start_time

        # Label remaining unlabeled samples using many-shot
        print("\nLabeling remaining samples with many-shot...")
        labeled_keys = set((s.question, s.choice) for s in self.labeled_dataset)
        unlabeled_remaining = [s for s in self.unlabeled_samples
                               if (s.question, s.choice) not in labeled_keys]

        for sample in unlabeled_remaining:
            predicted_label = self.assign_label(sample, self.labeled_dataset)
            labeled_sample = TruthfulQASample(
                question=sample.question,
                choice=sample.choice,
                ground_truth_label=sample.ground_truth_label,
                predicted_label=predicted_label,
                consistency_id=sample.consistency_id
            )
            self.labeled_dataset.append(labeled_sample)

        print(f"Labeled {len(unlabeled_remaining)} additional samples")

        print("="*80)
        print(f"\n{'FINAL RESULTS':^80}")
        print("="*80)
        print(f"  Initial Energy:        {initial_score:10.2f}")
        print(f"  Final Energy:          {final_score:10.2f}")
        print(f"  Energy Change:         {energy_change:10.2f} ({energy_change/initial_score:+.1%})" if initial_score != 0 else f"  Energy Change:         {energy_change:10.2f}")
        print(f"  Total Acceptances:     {total_acceptances:10d} / {num_iterations} ({total_acceptances/num_iterations:.1%})")
        print(f"  Final Dataset Size:    {len(self.labeled_dataset):10d}")
        print(f"  Total Time:            {total_time:10.1f}s")
        print("="*80 + "\n")

        # Add final results to log
        log_data["final"] = {
            "initial_score": initial_score,
            "final_score": final_score,
            "energy_change": energy_change,
            "total_acceptances": total_acceptances,
            "acceptance_rate": total_acceptances / num_iterations,
            "dataset_size": len(self.labeled_dataset),
            "final_labels": [
                {"question": s.question, "choice": s.choice, "label": s.predicted_label}
                for s in self.labeled_dataset
            ]
        }

        # Write log to file if specified
        if self.log_file:
            with open(self.log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
            print(f"Log written to {self.log_file}")

        return self.labeled_dataset


    def _create_proposal(self, sample: TruthfulQASample, label: bool) -> Tuple[List[TruthfulQASample], Dict[Tuple[str, str], int]]:
        """
        Create proposed dataset by adding/updating a sample with new label.

        Args:
            sample: Sample to add/update
            label: Label to assign

        Returns:
            Tuple of (proposed dataset, updated sample dict)
        """
        # Create new sample with updated label
        new_labeled_sample = TruthfulQASample(
            question=sample.question,
            choice=sample.choice,
            ground_truth_label=sample.ground_truth_label,
            predicted_label=label,
            consistency_id=sample.consistency_id
        )

        sample_key = (sample.question, sample.choice)

        if sample_key in self.sample_dict:
            # Replace existing
            proposed = self.labeled_dataset.copy()
            proposed[self.sample_dict[sample_key]] = new_labeled_sample
            proposed_dict = self.sample_dict.copy()
        else:
            # Add new
            proposed = self.labeled_dataset + [new_labeled_sample]
            proposed_dict = self.sample_dict.copy()
            proposed_dict[sample_key] = len(self.labeled_dataset)

        return proposed, proposed_dict
