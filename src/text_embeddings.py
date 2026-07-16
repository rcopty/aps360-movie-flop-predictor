"""Generate fixed embeddings from movie synopses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def clean_overviews(overviews: pd.Series) -> list[str]:
    """Replace missing synopsis values with empty strings."""
    cleaned = []
    for value in overviews:
        if pd.isna(value):
            cleaned.append("")
        else:
            cleaned.append(str(value).strip())
    return cleaned


def generate_synopsis_embeddings(
    overviews: pd.Series,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    """
    Convert movie overviews into fixed pretrained text embeddings.

    The pretrained encoder is used only for inference and is not fine-tuned.
    """
    texts = clean_overviews(overviews)

    encoder = SentenceTransformer(
        model_name,
        device=device,
    )

    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def save_embeddings(
    embeddings: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Save embeddings as a NumPy array."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    return output_path