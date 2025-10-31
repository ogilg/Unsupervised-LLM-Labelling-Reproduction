"""
Prompt formatting for TruthfulQA samples.
"""
from typing import List
from .data_models import TruthfulQASample


class PromptFormatter:
    """Formats TruthfulQA samples into prompts for language models."""

    def __init__(self, label_format: str = "True/False"):
        """
        Initialize the prompt formatter.

        Args:
            label_format: String indicating the format for labels (default: "True/False")
        """
        self.label_format = label_format

    def format_single(self, sample: TruthfulQASample, include_label: bool = False) -> str:
        """
        Format a single TruthfulQA sample into a prompt.

        Args:
            sample: The TruthfulQA sample to format
            include_label: Whether to include the predicted label in the output

        Returns:
            Formatted prompt string
        """
        prompt = f"Question: {sample.question}\nClaim: {sample.choice}\nI think this Claim is"

        if include_label and sample.predicted_label is not None:
            label_str = "True" if sample.predicted_label else "False"
            prompt += f" {label_str}"

        return prompt

    def format_icl(
        self,
        context_samples: List[TruthfulQASample],
        query_sample: TruthfulQASample
    ) -> str:
        """
        Format multiple samples for in-context learning (ICL).

        This is used for few-shot and many-shot prompting, where labeled examples
        are provided in the context before the query sample.

        Args:
            context_samples: List of labeled samples to include as context/examples
            query_sample: The sample to evaluate (without label in prompt)

        Returns:
            Formatted prompt with context examples followed by query
        """
        # Format all context examples with their labels
        context_prompts = [
            self.format_single(sample, include_label=True)
            for sample in context_samples
        ]

        # Format the query without label
        query_prompt = self.format_single(query_sample, include_label=False)

        # Combine with double newline separator
        all_prompts = context_prompts + [query_prompt]
        return "\n\n".join(all_prompts)

