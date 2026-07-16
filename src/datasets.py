"""PyTorch datasets for movie-flop prediction."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset

class MultiTaskMovieDataset(Dataset):
    """Dataset for flop classification and return-ratio regression."""

    def __init__(
        self,
        features: np.ndarray,
        flop_labels: np.ndarray,
        ratio_targets: np.ndarray,
    ) -> None:
        self.features = torch.as_tensor(
            features,
            dtype=torch.float32,
        )

        self.flop_labels = torch.as_tensor(
            flop_labels,
            dtype=torch.float32,
        )

        self.ratio_targets = torch.as_tensor(
            ratio_targets,
            dtype=torch.float32,
        )

        number_of_samples = len(self.flop_labels)

        if len(self.features) != number_of_samples:
            raise ValueError(
                "Features and flop labels must have the same length."
            )

        if len(self.ratio_targets) != number_of_samples:
            raise ValueError(
                "Ratio targets and flop labels must have the same length."
            )

    def __len__(self) -> int:
        return len(self.flop_labels)

    def __getitem__(self, index: int):
        return (
            self.features[index],
            self.flop_labels[index],
            self.ratio_targets[index],
        )

class TabularMovieDataset(Dataset):
    """Dataset containing structured movie features and binary labels."""
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.features = torch.as_tensor(
            features,
            dtype=torch.float32,
        )
        self.labels = torch.as_tensor(
            labels,
            dtype=torch.float32,
        )
        if len(self.features) != len(self.labels):
            raise ValueError(
                "Features and labels must contain the same number of samples."
            )
        
    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]
    

class MultimodalMovieDataset(Dataset):
    """Dataset containing tabular features, text embeddings, and labels."""

    def __init__(
        self,
        tabular_features: np.ndarray,
        text_embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.tabular_features = torch.as_tensor(
            tabular_features,
            dtype=torch.float32,
        )
        self.text_embeddings = torch.as_tensor(
            text_embeddings,
            dtype=torch.float32,
        )
        self.labels = torch.as_tensor(
            labels,
            dtype=torch.float32,
        )

        number_of_samples = len(self.labels)
        if len(self.tabular_features) != number_of_samples:
            raise ValueError(
                "Tabular features and labels must have the same length."
            )
        if len(self.text_embeddings) != number_of_samples:
            raise ValueError(
                "Text embeddings and labels must have the same length."
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            self.tabular_features[index],
            self.text_embeddings[index],
            self.labels[index],
        )