"""Neural-network architectures for movie-flop prediction."""

from __future__ import annotations
import torch
from torch import nn


class TabularMLP(nn.Module):
    """Small MLP for structured pre-release movie features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int = 64,
        hidden_dim_2: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Return one raw classification logit per movie.
        """
        return self.network(features).squeeze(1)
    
class TextMLP(nn.Module):
    """Small classifier trained on frozen synopsis embeddings."""

    def __init__(
        self,
        embedding_dim: int = 384,
        hidden_dim_1: int = 32,
        hidden_dim_2: int = 8,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, 1),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Return one raw flop-classification logit per movie."""
        return self.network(embeddings).squeeze(1)
    

class MultimodalClassifier(nn.Module):
    """Combine structured movie features with frozen synopsis embeddings."""

    def __init__(
        self,
        tabular_dim: int,
        text_dim: int = 384,
        tabular_hidden_dim: int = 32,
        tabular_output_dim: int = 16,
        text_hidden_dim: int = 32,
        text_output_dim: int = 8,
        fusion_hidden_dim: int = 16,
        tabular_dropout: float = 0.3,
        text_dropout: float = 0.5,
        fusion_dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, tabular_hidden_dim),
            nn.ReLU(),
            nn.Dropout(tabular_dropout),
            nn.Linear(tabular_hidden_dim, tabular_output_dim),
            nn.ReLU(),
        )

        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, text_hidden_dim),
            nn.ReLU(),
            nn.Dropout(text_dropout),
            nn.Linear(text_hidden_dim, text_output_dim),
            nn.ReLU(),
        )

        fusion_input_dim = tabular_output_dim + text_output_dim

        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(
        self,
        tabular_features: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        tabular_representation = self.tabular_encoder(
            tabular_features
        )

        text_representation = self.text_encoder(
            text_embeddings
        )

        combined_representation = torch.cat(
            [
                tabular_representation,
                text_representation,
            ],
            dim=1,
        )

        return self.fusion_network(
            combined_representation
        ).squeeze(1)
    
class MultiTaskTabularMLP(nn.Module):
    """Jointly predict flop probability and financial return."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, int] = (64, 32),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        hidden_dim_1, hidden_dim_2 = hidden_dims

        self.shared_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classification_head = nn.Linear(
            hidden_dim_2,
            1,
        )

        self.regression_head = nn.Linear(
            hidden_dim_2,
            1,
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.shared_network(features)

        flop_logits = self.classification_head(
            representation
        ).squeeze(1)

        predicted_ratio = self.regression_head(
            representation
        ).squeeze(1)

        return flop_logits, predicted_ratio