"""Evaluate the best trained MNIST SNN checkpoint on the complete test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from software.paths import DATA_DIR, DEFAULT_CHECKPOINT
from software.snn.dataset import get_mnist_loaders
from software.snn.model import MNISTSNN, count_output_spikes


def evaluate(
    model: MNISTSNN,
    loader: DataLoader,
    device: torch.device,
    show_examples: int = 5,
) -> tuple[int, int]:
    """Run inference, print selected examples, and return correct/total."""
    if show_examples < 0:
        raise ValueError("show_examples must be non-negative")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            spike_record = model(images)
            spike_counts = count_output_spikes(spike_record)
            predictions = spike_counts.argmax(dim=1)

            samples_to_show = min(show_examples - total, labels.size(0))
            for index in range(max(samples_to_show, 0)):
                counts = spike_counts[index].to(torch.int64).cpu().tolist()
                print(
                    f"sample {total + index}: "
                    f"true={int(labels[index])} "
                    f"predicted={int(predictions[index])} "
                    f"spike_counts={counts}"
                )

            correct += (predictions.cpu() == labels).sum().item()
            total += labels.numel()

    if total == 0:
        raise ValueError("test loader produced no samples")
    return correct, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="best-validation checkpoint produced by train.py",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--examples",
        type=int,
        default=5,
        help="number of initial test examples to print with their class counts",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.examples < 0:
        parser.error("--examples must be non-negative")
    if not args.checkpoint.is_file():
        parser.error(
            f"checkpoint not found: {args.checkpoint}; run train.py first "
            "to create the best-validation checkpoint"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    required_keys = {
        "model_state_dict",
        "hidden_size",
        "beta",
        "threshold",
        "num_steps",
    }
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        parser.error(
            "checkpoint is missing required model configuration: "
            + ", ".join(sorted(missing_keys))
        )

    model = MNISTSNN(
        hidden_size=checkpoint["hidden_size"],
        beta=checkpoint["beta"],
        threshold=checkpoint["threshold"],
        num_steps=checkpoint["num_steps"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, test_loader = get_mnist_loaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
    )
    print(
        f"device={device} test_samples={len(test_loader.dataset)} "
        f"num_steps={model.num_steps} batch_size={args.batch_size}",
        flush=True,
    )
    correct, total = evaluate(
        model,
        test_loader,
        device,
        show_examples=args.examples,
    )
    accuracy = 100.0 * correct / total
    print(f"Correct predictions: {correct}")
    print(f"Test samples: {total}")
    print(f"Floating-point accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
