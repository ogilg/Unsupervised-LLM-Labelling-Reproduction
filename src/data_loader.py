"""
Data loading utilities for TruthfulQA dataset.
"""
import json
from typing import List
from pathlib import Path
from .data_models import TruthfulQASample


class TruthfulQALoader:
    """Loader for TruthfulQA dataset."""

    @staticmethod
    def load_from_json(file_path: str, max_samples: int = None) -> List[TruthfulQASample]:
        """
        Load TruthfulQA samples from JSON file.

        Args:
            file_path: Path to the JSON file
            max_samples: Maximum number of samples to load. If None, load all samples.

        Returns:
            List of TruthfulQASample objects
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Limit data if max_samples is specified
        if max_samples is not None:
            data = data[:max_samples]

        samples = []
        for item in data:
            sample = TruthfulQASample(
                question=item['question'],
                choice=item['choice'],
                ground_truth_label=bool(item['label']) if 'label' in item else None,
                consistency_id=item.get('consistency_id')
            )
            samples.append(sample)

        return samples

    @staticmethod
    def load_train_test(
        data_dir: str = "data",
        max_train: int = None,
        max_test: int = None
    ):
        """
        Load both train and test splits.

        Args:
            data_dir: Directory containing the data files
            max_train: Maximum number of train samples to load. If None, load all.
            max_test: Maximum number of test samples to load. If None, load all.

        Returns:
            Tuple of (train_samples, test_samples)
        """
        data_path = Path(data_dir)
        train_samples = TruthfulQALoader.load_from_json(
            data_path / "truthfulqa_train.json",
            max_samples=max_train
        )
        test_samples = TruthfulQALoader.load_from_json(
            data_path / "truthfulqa_test.json",
            max_samples=max_test
        )
        return train_samples, test_samples


def load_truthfulqa_data(split: str = 'train', data_dir: str = "data", max_samples: int = None):
    """
    Convenience function to load TruthfulQA data.

    Args:
        split: 'train' or 'test'
        data_dir: Directory containing the data files
        max_samples: Maximum number of samples to load

    Returns:
        List of TruthfulQASample objects
    """
    data_path = Path(data_dir)
    filename = f"truthfulqa_{split}.json"
    return TruthfulQALoader.load_from_json(data_path / filename, max_samples=max_samples)
