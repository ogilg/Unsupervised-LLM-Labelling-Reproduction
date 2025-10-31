"""
Data models for TruthfulQA dataset.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class TruthfulQASample:
    """TruthfulQA sample with both predicted and ground truth labels."""
    question: str
    choice: str
    ground_truth_label: Optional[bool] = None
    predicted_label: Optional[bool] = None
    consistency_id: Optional[int] = None
