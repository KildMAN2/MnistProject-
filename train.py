"""Train the MNIST SNN with surrogate-gradient backpropagation through time."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from dataset import get_mnist_loaders
from model import (
    DEFAULT_BETA,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_NUM_STEPS,
    DEFAULT_THRESHOLD,
    MNISTSNN,
    count_output_spikes,
    predict_from_spikes,
)

VALIDATION_SIZE = 5_000


def train_one_epoch(
    model: MNISTSNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    """Train one epoch; return sample-weighted loss and online accuracy."""
    model.train()
    loss_total = 0.0
    correct = 0
    samples = 0
    total_batches = len(loader)

    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        output_spikes = model(images)
        spike_counts = count_output_spikes(output_spikes)
        loss = loss_function(spike_counts, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        loss_total += loss.detach().item() * batch_size
        predictions = spike_counts.detach().argmax(dim=1)
        correct += (predictions == labels).sum().item()
        samples += batch_size

        if batch_index == 1 or batch_index % 100 == 0 or batch_index == total_batches:
            print(
                f"epoch {epoch}/{total_epochs} "
                f"batch {batch_index}/{total_batches} loss={loss.item():.4f}",
                flush=True,
            )

    if samples == 0:
        raise ValueError("training loader produced no samples")
    return loss_total / samples, correct / samples


@torch.no_grad()
def evaluate_accuracy(
    model: MNISTSNN,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate classification accuracy without updating model parameters."""
    model.eval()
    correct = 0
    samples = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        predictions = predict_from_spikes(model(images))
        correct += (predictions == labels).sum().item()
        samples += labels.numel()

    if samples == 0:
        raise ValueError("evaluation loader produced no samples")
    return correct / samples


def save_checkpoint(
    output_path: Path,
    model: MNISTSNN,
    epoch: int,
    train_loss: float,
    train_accuracy: float,
    validation_accuracy: float,
) -> None:
    """Atomically save model weights and their inference configuration."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "hidden_size": model.hidden_size,
        "num_steps": model.num_steps,
        "beta": model.beta,
        "threshold": model.threshold,
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "validation_accuracy": validation_accuracy,
    }
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    try:
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output", type=Path, default=Path("mnist_snn_static.pt")
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.num_steps < 1:
        parser.error("--num-steps must be at least 1")
    if args.hidden_size < 1:
        parser.error("--hidden-size must be at least 1")
    if not 0.0 <= args.beta <= 1.0:
        parser.error("--beta must be between 0 and 1")
    if args.threshold <= 0.0:
        parser.error("--threshold must be positive")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_train_loader, test_loader = get_mnist_loaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
    )
    if VALIDATION_SIZE >= len(full_train_loader.dataset):
        parser.error("training dataset is too small to create a validation split")

    training_data, validation_data = random_split(
        full_train_loader.dataset,
        [len(full_train_loader.dataset) - VALIDATION_SIZE, VALIDATION_SIZE],
        generator=torch.Generator().manual_seed(0),
    )
    train_loader = DataLoader(
        training_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(0),
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = MNISTSNN(
        hidden_size=args.hidden_size,
        beta=args.beta,
        threshold=args.threshold,
        num_steps=args.num_steps,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.CrossEntropyLoss()
    best_validation_accuracy = -1.0

    print(
        f"device={device} train={len(training_data)} "
        f"validation={len(validation_data)} test={len(test_loader.dataset)} "
        f"batch_size={args.batch_size} num_steps={model.num_steps} "
        f"hidden_size={model.hidden_size} lr={args.learning_rate}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            device,
            epoch,
            args.epochs,
        )
        validation_accuracy = evaluate_accuracy(
            model,
            validation_loader,
            device,
        )
        is_best = validation_accuracy > best_validation_accuracy
        if is_best:
            best_validation_accuracy = validation_accuracy
            save_checkpoint(
                args.output,
                model,
                epoch,
                train_loss,
                train_accuracy,
                validation_accuracy,
            )

        print(
            f"epoch {epoch}/{args.epochs} complete "
            f"train_loss={train_loss:.4f} "
            f"train_accuracy={train_accuracy:.4f} "
            f"validation_accuracy={validation_accuracy:.4f}"
            f"{' best_checkpoint_saved' if is_best else ''}",
            flush=True,
        )

    checkpoint = torch.load(args.output, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_accuracy = evaluate_accuracy(model, test_loader, device)
    test_count = round(test_accuracy * len(test_loader.dataset))
    print(
        f"best epoch={checkpoint['epoch']} "
        f"best validation accuracy={checkpoint['validation_accuracy']:.4f} "
        f"floating-point test accuracy={test_accuracy:.4f} "
        f"({test_count}/{len(test_loader.dataset)})",
        flush=True,
    )
    print(f"Best checkpoint saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
