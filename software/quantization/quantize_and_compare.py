"""Export int16 SNN weights and compare floating-point and integer inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from software.paths import DATA_DIR, DEFAULT_CHECKPOINT, INT16_DIR
from software.quantization.quantization import (
    Int16SNN,
    export_quantized_model,
    quantized_spike_counts,
)
from software.snn.dataset import get_mnist_loaders
from software.snn.model import MNISTSNN, count_output_spikes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--export-dir", type=Path, default=INT16_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.examples < 0:
        parser.error("--examples must be non-negative")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}; run train.py first")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = MNISTSNN(
        hidden_size=checkpoint["hidden_size"],
        beta=checkpoint["beta"],
        threshold=checkpoint["threshold"],
        num_steps=checkpoint["num_steps"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    metadata = export_quantized_model(model, args.export_dir)
    quantized_model = Int16SNN(args.export_dir)
    print(f"Exported quantized weights and integer biases to {args.export_dir}")
    for name, layer in metadata["layers"].items():
        print(
            f"{name}: int16 range=[{layer['integer_weight_min']}, "
            f"{layer['integer_weight_max']}] scale={layer['weight_scale']:.12g} "
            f"shape={layer['weight_shape']} bias int32 scale={layer['bias_scale']:.12g}"
        )
        for sample in layer["weight_sample_comparison"]:
            print(
                f"  weight float={sample['float']:.8f} "
                f"int16={sample['int16']} "
                f"reconstructed={sample['reconstructed']:.8f}"
            )

    _, test_loader = get_mnist_loaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
    )
    float_correct = 0
    integer_correct = 0
    prediction_agreement = 0
    total = 0
    shown = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images_device = images.to(device)
            float_spikes = model(images_device)
            float_counts = count_output_spikes(float_spikes).cpu()
            float_predictions = float_counts.argmax(dim=1)

            integer_spikes = quantized_model(images)
            integer_counts = quantized_spike_counts(integer_spikes)
            integer_predictions = integer_counts.argmax(dim=1)

            if shown < args.examples:
                for index in range(min(args.examples - shown, labels.size(0))):
                    print(
                        f"sample {shown + index}: true={int(labels[index])} "
                        f"float={int(float_predictions[index])} "
                        f"int16={int(integer_predictions[index])} "
                        f"int16_counts={integer_counts[index].tolist()}"
                    )
                shown += min(args.examples - shown, labels.size(0))

            float_correct += (float_predictions == labels).sum().item()
            integer_correct += (integer_predictions == labels).sum().item()
            prediction_agreement += (
                float_predictions == integer_predictions
            ).sum().item()
            total += labels.numel()

    float_accuracy = 100.0 * float_correct / total
    integer_accuracy = 100.0 * integer_correct / total
    print(f"Floating-point accuracy: {float_accuracy:.2f}%")
    print(f"Int16-weight accuracy:   {integer_accuracy:.2f}%")
    print(f"Accuracy loss:           {float_accuracy - integer_accuracy:.2f}%")
    print(f"Prediction agreement:    {100.0 * prediction_agreement / total:.2f}%")
    print(f"Float correct: {float_correct}/{total}; int16 correct: {integer_correct}/{total}")


if __name__ == "__main__":
    main()
