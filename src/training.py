"""Training utilities for PyTorch movie-flop models."""

from __future__ import annotations
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import random

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch and return the mean loss."""
    model.train()
    total_loss = 0.0
    total_examples = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(features)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
    return total_loss / total_examples


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    """Calculate mean loss without changing model parameters."""

    model.eval()
    total_loss = 0.0
    total_examples = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        loss = loss_function(logits, labels)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
    return total_loss / total_examples


def fit_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
    max_epochs: int = 100,
    patience: int = 12,
) -> dict:
    """Train with early stopping based on validation loss."""
    history = {
        "train_loss": [],
        "validation_loss": [],
    }

    best_validation_loss = np.inf
    best_state = None
    epochs_without_improvement = 0
    model.to(device)

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device)
        validation_loss = evaluate_loss(
            model=model,
            loader=validation_loader,
            loss_function=loss_function,
            device=device)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")
    model.load_state_dict(best_state)
    history["epochs_trained"] = len(history["train_loss"])
    history["best_validation_loss"] = best_validation_loss
    return history


@torch.no_grad()
def predict_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return binary predictions and flop probabilities."""
    model.eval()
    model.to(device)
    all_probabilities = []
    for features, _ in loader:
        features = features.to(device)
        logits = model(features)
        probabilities = torch.sigmoid(logits)
        all_probabilities.append(
            probabilities.cpu().numpy())
    probabilities = np.concatenate(all_probabilities)
    predictions = (probabilities >= threshold).astype(int)
    return predictions, probabilities

def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable model parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad)


## Multimodal Model

def train_multimodal_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    """Train a multimodal classifier for one epoch."""

    model.train()

    total_loss = 0.0
    total_examples = 0

    for tabular_features, text_embeddings, labels in loader:
        tabular_features = tabular_features.to(device)
        text_embeddings = text_embeddings.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(
            tabular_features,
            text_embeddings,
        )

        loss = loss_function(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples


@torch.no_grad()
def evaluate_multimodal_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    """Calculate multimodal validation loss."""

    model.eval()

    total_loss = 0.0
    total_examples = 0

    for tabular_features, text_embeddings, labels in loader:
        tabular_features = tabular_features.to(device)
        text_embeddings = text_embeddings.to(device)
        labels = labels.to(device)

        logits = model(
            tabular_features,
            text_embeddings,
        )

        loss = loss_function(logits, labels)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples


def fit_multimodal_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
    max_epochs: int = 100,
    patience: int = 12,
) -> dict:
    """Train a multimodal classifier with early stopping."""

    history = {
        "train_loss": [],
        "validation_loss": [],
    }

    best_validation_loss = np.inf
    best_state = None
    epochs_without_improvement = 0

    model.to(device)

    for _ in range(max_epochs):
        train_loss = train_multimodal_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
        )

        validation_loss = evaluate_multimodal_loss(
            model=model,
            loader=validation_loader,
            loss_function=loss_function,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("No valid model state was produced.")

    model.load_state_dict(best_state)

    history["epochs_trained"] = len(history["train_loss"])
    history["best_validation_loss"] = best_validation_loss

    return history


@torch.no_grad()
def predict_multimodal_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return multimodal predictions and flop probabilities."""

    model.eval()
    model.to(device)

    all_probabilities = []

    for tabular_features, text_embeddings, _ in loader:
        tabular_features = tabular_features.to(device)
        text_embeddings = text_embeddings.to(device)

        logits = model(
            tabular_features,
            text_embeddings,
        )

        probabilities = torch.sigmoid(logits)

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

    probabilities = np.concatenate(all_probabilities)
    predictions = (probabilities >= threshold).astype(int)

    return predictions, probabilities

def train_multitask_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    classification_loss_function: nn.Module,
    regression_loss_function: nn.Module,
    regression_weight: float,
    device: torch.device,
) -> dict[str, float]:
    """Train a multi-task model for one epoch."""

    model.train()

    total_loss_sum = 0.0
    classification_loss_sum = 0.0
    regression_loss_sum = 0.0
    total_examples = 0

    for features, flop_labels, ratio_targets in loader:
        features = features.to(device)
        flop_labels = flop_labels.to(device)
        ratio_targets = ratio_targets.to(device)

        optimizer.zero_grad()

        flop_logits, ratio_predictions = model(features)

        classification_loss = classification_loss_function(
            flop_logits,
            flop_labels,
        )

        regression_loss = regression_loss_function(
            ratio_predictions,
            ratio_targets,
        )

        total_loss = (
            classification_loss
            + regression_weight * regression_loss
        )

        total_loss.backward()
        optimizer.step()

        batch_size = flop_labels.size(0)

        total_loss_sum += total_loss.item() * batch_size
        classification_loss_sum += (
            classification_loss.item() * batch_size
        )
        regression_loss_sum += (
            regression_loss.item() * batch_size
        )

        total_examples += batch_size

    return {
        "total_loss": total_loss_sum / total_examples,
        "classification_loss": (
            classification_loss_sum / total_examples
        ),
        "regression_loss": (
            regression_loss_sum / total_examples
        ),
    }

@torch.no_grad()
def evaluate_multitask_loss(
    model: nn.Module,
    loader: DataLoader,
    classification_loss_function: nn.Module,
    regression_loss_function: nn.Module,
    regression_weight: float,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate multi-task validation losses."""

    model.eval()

    total_loss_sum = 0.0
    classification_loss_sum = 0.0
    regression_loss_sum = 0.0
    total_examples = 0

    for features, flop_labels, ratio_targets in loader:
        features = features.to(device)
        flop_labels = flop_labels.to(device)
        ratio_targets = ratio_targets.to(device)

        flop_logits, ratio_predictions = model(features)

        classification_loss = classification_loss_function(
            flop_logits,
            flop_labels,
        )

        regression_loss = regression_loss_function(
            ratio_predictions,
            ratio_targets,
        )

        total_loss = (
            classification_loss
            + regression_weight * regression_loss
        )

        batch_size = flop_labels.size(0)

        total_loss_sum += total_loss.item() * batch_size
        classification_loss_sum += (
            classification_loss.item() * batch_size
        )
        regression_loss_sum += (
            regression_loss.item() * batch_size
        )

        total_examples += batch_size

    return {
        "total_loss": total_loss_sum / total_examples,
        "classification_loss": (
            classification_loss_sum / total_examples
        ),
        "regression_loss": (
            regression_loss_sum / total_examples
        ),
    }

def fit_multitask_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    classification_loss_function: nn.Module,
    regression_loss_function: nn.Module,
    regression_weight: float,
    device: torch.device,
    max_epochs: int = 100,
    patience: int = 12,
) -> dict:
    """Train a multi-task model with early stopping."""

    history = {
        "train_total_loss": [],
        "validation_total_loss": [],
        "train_classification_loss": [],
        "validation_classification_loss": [],
        "train_regression_loss": [],
        "validation_regression_loss": [],
    }

    best_validation_loss = np.inf
    best_state = None
    epochs_without_improvement = 0

    model.to(device)

    for _ in range(max_epochs):
        train_losses = train_multitask_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            classification_loss_function=(
                classification_loss_function
            ),
            regression_loss_function=regression_loss_function,
            regression_weight=regression_weight,
            device=device,
        )

        validation_losses = evaluate_multitask_loss(
            model=model,
            loader=validation_loader,
            classification_loss_function=(
                classification_loss_function
            ),
            regression_loss_function=regression_loss_function,
            regression_weight=regression_weight,
            device=device,
        )

        history["train_total_loss"].append(
            train_losses["total_loss"]
        )
        history["validation_total_loss"].append(
            validation_losses["total_loss"]
        )

        history["train_classification_loss"].append(
            train_losses["classification_loss"]
        )
        history["validation_classification_loss"].append(
            validation_losses["classification_loss"]
        )

        history["train_regression_loss"].append(
            train_losses["regression_loss"]
        )
        history["validation_regression_loss"].append(
            validation_losses["regression_loss"]
        )

        current_validation_loss = validation_losses[
            "total_loss"
        ]

        if current_validation_loss < best_validation_loss:
            best_validation_loss = current_validation_loss
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("No valid model state was saved.")

    model.load_state_dict(best_state)

    history["epochs_trained"] = len(
        history["train_total_loss"]
    )
    history["best_validation_loss"] = (
        best_validation_loss
    )

    return history

@torch.no_grad()
def predict_multitask_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return class predictions, probabilities, and ratio predictions."""

    model.eval()
    model.to(device)

    probability_batches = []
    ratio_batches = []

    for features, _, _ in loader:
        features = features.to(device)

        flop_logits, ratio_predictions = model(features)

        probabilities = torch.sigmoid(flop_logits)

        probability_batches.append(
            probabilities.cpu().numpy()
        )

        ratio_batches.append(
            ratio_predictions.cpu().numpy()
        )

    probabilities = np.concatenate(
        probability_batches
    )

    ratio_predictions = np.concatenate(
        ratio_batches
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return predictions, probabilities, ratio_predictions