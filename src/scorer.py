"""
Mutual predictability scorer for ICM algorithm.
"""
import math
from typing import List, Tuple
from .data_models import TruthfulQASample
from .prompt_formatter import PromptFormatter
from .hyperbolic_wrapper import HyperbolicClient

TEMPERATURE = 0.0


class MutualPredictabilityScorer:
    """Computes mutual predictability score for labeled samples."""

    def __init__(self, model_client: HyperbolicClient):
        self.client = model_client
        self.formatter = PromptFormatter()
        self.cache = {}  # Cache: prompt -> logprobs_data
        self.cache_hits = 0
        self.cache_misses = 0

    def clear_cache(self):
        """Clear the logprobs cache."""
        self.cache = {}

    def get_cache_stats(self):
        """Get cache hit/miss statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "total": total,
            "hit_rate": hit_rate
        }

    def get_both_logprobs(self, sample: TruthfulQASample, other_samples: List[TruthfulQASample]) -> Tuple[float, float]:
        """Get log probabilities for both True and False labels in one API call.

        Returns:
            Tuple of (true_logprob, false_logprob)
        """
        prompt = self.formatter.format_icl(other_samples, sample)

        # Check cache first
        if prompt in self.cache:
            logprobs_data = self.cache[prompt]
            self.cache_hits += 1
        else:
            # Make API call and cache result
            response = self.client.generate(prompt, max_tokens=1, temperature=TEMPERATURE, logprobs=5)
            logprobs_data = self.client.get_logprobs(response)
            self.cache[prompt] = logprobs_data
            self.cache_misses += 1

        true_logprob = float('-inf')
        false_logprob = float('-inf')

        if logprobs_data and logprobs_data.top_logprobs:
            top_logprobs = logprobs_data.top_logprobs[0]

            for token_str, logprob_val in top_logprobs.items():
                token = token_str.strip()
                if token == "True":
                    true_logprob = logprob_val
                elif token == "False":
                    false_logprob = logprob_val

        return true_logprob, false_logprob

    def get_logprob_for_label(self, sample: TruthfulQASample, other_samples: List[TruthfulQASample], label: bool) -> float:
        """Get log probability of a specific label conditioned on other samples."""
        true_logprob, false_logprob = self.get_both_logprobs(sample, other_samples)
        return true_logprob if label else false_logprob

    def get_label_logprob(self, sample: TruthfulQASample, other_samples: List[TruthfulQASample]) -> float:
        """Get log probability of sample's predicted label conditioned on other samples."""
        return self.get_logprob_for_label(sample, other_samples, sample.predicted_label)

    def compute_score(self, labeled_samples: List[TruthfulQASample]) -> float:
        r"""
        Compute mutual predictability score: MP(L) = Σ log P(y_i | x_i, L \ {(x_i, y_i)})

        Args:
            labeled_samples: List of samples with labels

        Returns:
            Sum of log probabilities
        """
        if len(labeled_samples) < 2:
            return 0.0

        total_score = 0.0

        for i, target_sample in enumerate(labeled_samples):
            # leave one out
            other_samples = labeled_samples[:i] + labeled_samples[i+1:]
            # get logprob of label
            logprob = self.get_label_logprob(target_sample, other_samples)
            total_score += logprob

        return total_score
