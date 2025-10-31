"""
Internal Coherence Maximization (ICM) Algorithm with Fixed Context Size.

This variant uses a fixed context window size C and reservoir sampling to update
context windows when accepting new samples, enabling better cache utilization.
"""
import random
from typing import List, Optional
from .icm_algorithm import ICMAlgorithm
from .data_models import TruthfulQASample


class ICMFixedContext(ICMAlgorithm):
    """ICM with fixed context size for improved caching efficiency."""

    def __init__(
        self,
        unlabeled_samples: List[TruthfulQASample],
        model_client,
        context_size: int = 20,
        initial_temp: float = 10.0,
        final_temp: float = 0.01,
        cooling_rate: float = 0.99,
        log_file: Optional[str] = None
    ):
        super().__init__(
            unlabeled_samples=unlabeled_samples,
            model_client=model_client,
            initial_temp=initial_temp,
            final_temp=final_temp,
            cooling_rate=cooling_rate,
            log_file=log_file
        )
        self.context_size = context_size
        # For each sample index, store indices of samples to use as context
        self.context_indices = {}

    def initialize_random_labels(self, k: int = 8) -> List[TruthfulQASample]:
        """Initialize with random labels and set up context indices."""
        labeled = super().initialize_random_labels(k)

        # Initialize context for each sample
        self.context_indices = {}
        for i in range(len(labeled)):
            # Context: randomly sample other samples (up to context_size)
            other_indices = [j for j in range(len(labeled)) if j != i]
            if len(other_indices) > self.context_size:
                self.context_indices[i] = random.sample(other_indices, self.context_size)
            else:
                self.context_indices[i] = other_indices

        return labeled

    def get_context_for_sample(self, sample_idx: int) -> List[TruthfulQASample]:
        """Get the fixed-size context for a sample."""
        context_idx = self.context_indices.get(sample_idx, [])
        return [self.labeled_dataset[i] for i in context_idx if i < len(self.labeled_dataset)]

    def compute_score(self, labeled_samples: List[TruthfulQASample]) -> float:
        """Compute score using fixed context windows instead of leave-one-out."""
        if len(labeled_samples) < 2:
            return 0.0

        total_score = 0.0
        for i in range(len(labeled_samples)):
            context = self.get_context_for_sample(i)
            target_sample = labeled_samples[i]
            logprob = self.scorer.get_logprob_for_label(
                target_sample, context, target_sample.predicted_label
            )
            total_score += logprob

        return total_score

    def update_context_windows(self, new_sample_idx: int):
        """
        Update context windows using reservoir sampling when accepting a new sample.

        For each existing sample's context window:
        - With probability C/N, include the new sample
        - If included, randomly replace one existing sample in the context
        """
        N = len(self.labeled_dataset)
        C = self.context_size

        if N <= 1:
            return

        # Initialize context for the new sample
        other_indices = [i for i in range(N) if i != new_sample_idx]
        if len(other_indices) > C:
            other_indices = random.sample(other_indices, C)
        self.context_indices[new_sample_idx] = other_indices

        # Update existing samples' contexts with reservoir sampling
        for i in range(N):
            if i == new_sample_idx:
                continue

            # With probability C/N, include new sample in this context
            if random.random() < C / N:
                current_context = self.context_indices.get(i, [])

                if len(current_context) < C:
                    # Context not full, just add
                    current_context.append(new_sample_idx)
                else:
                    # Context full, replace random sample
                    replace_idx = random.randint(0, len(current_context) - 1)
                    current_context[replace_idx] = new_sample_idx

                self.context_indices[i] = current_context

    def run(self, num_iterations: int, initial_k: int = 8) -> List[TruthfulQASample]:
        """Run ICM with fixed context size."""
        import time
        start_time = time.time()
        
        log_data = {
            "config": {
                "num_iterations": num_iterations,
                "initial_k": initial_k,
                "context_size": self.context_size,
                "initial_temp": self.initial_temp,
                "final_temp": self.final_temp,
                "cooling_rate": self.cooling_rate,
                "num_unlabeled_samples": len(self.unlabeled_samples)
            },
            "iterations": []
        }

        # Initialize
        self.labeled_dataset = self.initialize_random_labels(initial_k)
        print(f"Initialized with {initial_k} random labels")

        initial_labels = [
            {"question": s.question, "choice": s.choice, "label": s.predicted_label}
            for s in self.labeled_dataset
        ]
        log_data["initialization"] = initial_labels

        # Monitoring
        total_acceptances = 0
        recent_acceptances = []
        current_score = self.compute_score(self.labeled_dataset)
        initial_score = current_score

        print("\n" + "="*90)
        print(f"{'Iter':>6} | {'Temp':>8} | {'Energy':>10} | {'Delta':>8} | {'Accept':>6} | {'Acc Rate':>8} | {'Size':>5} | {'Time':>7} | {'Cache':>7}")
        print("="*90)

        # Main loop
        checkpoint_start = time.time()
        for iteration in range(num_iterations):
            iter_start = time.time()
            temperature = self.update_temperature(iteration)
            target_sample = self.sample_example()
            sample_key = (target_sample.question, target_sample.choice)
            is_update = sample_key in self.sample_dict

            # Get context for labeling
            if is_update:
                sample_idx = self.sample_dict[sample_key]
                context = self.get_context_for_sample(sample_idx)
            else:
                # For new sample, use random subset as context
                if len(self.labeled_dataset) <= self.context_size:
                    context = self.labeled_dataset[:]
                else:
                    context = random.sample(self.labeled_dataset, self.context_size)

            new_label = self.assign_label(target_sample, context)

            # Create proposal
            proposed_dataset, proposed_dict = self._create_proposal(target_sample, new_label)

            # Save old state
            old_dataset = self.labeled_dataset
            old_dict = self.sample_dict
            old_context_indices = self.context_indices.copy()

            # Apply proposal temporarily
            self.labeled_dataset = proposed_dataset
            self.sample_dict = proposed_dict

            # Update context windows if adding new sample
            if not is_update:
                new_idx = len(old_dataset)
                self.update_context_windows(new_idx)

            # Compute scores
            proposed_score = self.compute_score(self.labeled_dataset)
            delta = proposed_score - current_score

            # Accept/reject
            accepted = self.accept_proposal(delta, temperature)
            if accepted:
                current_score = proposed_score
                total_acceptances += 1
            else:
                # Restore old state
                self.labeled_dataset = old_dataset
                self.sample_dict = old_dict
                self.context_indices = old_context_indices

            cumulative_rate = total_acceptances / (iteration + 1)

            iter_time = time.time() - iter_start
            cache_stats = self.scorer.get_cache_stats()

            iter_log = {
                "iteration": iteration,
                "temperature": temperature,
                "current_score": current_score,
                "proposed_score": proposed_score,
                "accepted": accepted,
                "dataset_size": len(self.labeled_dataset),
                "cumulative_acceptance_rate": cumulative_rate,
                "iter_time": iter_time,
                "cache_hits": cache_stats["hits"],
                "cache_hit_rate": cache_stats["hit_rate"],
                "is_update": is_update
            }
            log_data["iterations"].append(iter_log)

            if iteration % 10 == 0 or iteration < 5:
                accept_symbol = "✓" if accepted else "✗"
                checkpoint_time = time.time() - checkpoint_start
                num_iters_since_checkpoint = 10 if iteration >= 10 else iteration + 1
                avg_time = checkpoint_time / num_iters_since_checkpoint
                print(f"{iteration:6d} | {temperature:8.3f} | {current_score:10.2f} | {delta:8.2f} | "
                      f"{accept_symbol:>6} | {cumulative_rate:7.1%} | {len(self.labeled_dataset):5d} | "
                      f"{avg_time:5.2f}s | {cache_stats['hit_rate']:.1%}")
                checkpoint_start = time.time()

        final_score = self.compute_score(self.labeled_dataset)
        energy_change = final_score - initial_score

        # Label remaining unlabeled samples using many-shot
        print("\nLabeling remaining samples with many-shot...")
        labeled_keys = set((s.question, s.choice) for s in self.labeled_dataset)
        unlabeled_remaining = [s for s in self.unlabeled_samples
                               if (s.question, s.choice) not in labeled_keys]

        for sample in unlabeled_remaining:
            # Use random subset as context for labeling
            if len(self.labeled_dataset) <= self.context_size:
                context = self.labeled_dataset[:]
            else:
                context = random.sample(self.labeled_dataset, self.context_size)

            predicted_label = self.assign_label(sample, context)
            labeled_sample = TruthfulQASample(
                question=sample.question,
                choice=sample.choice,
                ground_truth_label=sample.ground_truth_label,
                predicted_label=predicted_label,
                consistency_id=sample.consistency_id
            )
            self.labeled_dataset.append(labeled_sample)

        print(f"Labeled {len(unlabeled_remaining)} additional samples")

        # Calculate accuracy and total time
        total_time = time.time() - start_time
        correct = sum(1 for s in self.labeled_dataset if s.predicted_label == s.ground_truth_label)
        accuracy = correct / len(self.labeled_dataset) if self.labeled_dataset else 0.0

        print("="*80)
        print(f"\n{'FINAL RESULTS':^80}")
        print("="*80)
        print(f"  Initial Energy:        {initial_score:10.2f}")
        print(f"  Final Energy:          {final_score:10.2f}")
        print(f"  Energy Change:         {energy_change:10.2f} ({energy_change/initial_score:+.1%})" if initial_score != 0 else f"  Energy Change:         {energy_change:10.2f}")
        print(f"  Total Acceptances:     {total_acceptances:10d} / {num_iterations} ({total_acceptances/num_iterations:.1%})")
        print(f"  Final Dataset Size:    {len(self.labeled_dataset):10d}")
        print(f"  Context Size:          {self.context_size:10d}")
        print(f"  Accuracy:              {accuracy:10.1%} ({correct}/{len(self.labeled_dataset)})")
        print(f"  Total Time:            {total_time:10.1f}s")
        print("="*80 + "\n")

        log_data["final"] = {
            "initial_score": initial_score,
            "final_score": final_score,
            "energy_change": energy_change,
            "total_acceptances": total_acceptances,
            "acceptance_rate": total_acceptances / num_iterations,
            "dataset_size": len(self.labeled_dataset),
            "accuracy": accuracy,
            "total_time": total_time
        }

        if self.log_file:
            import json
            with open(self.log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
            print(f"Log written to {self.log_file}")

        return self.labeled_dataset
