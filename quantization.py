"""Per-layer signed-int16 weight export and explicit integer SNN inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

INT16_MIN = -32768
INT16_MAX = 32767
BIAS_DTYPE = np.int32
PIXEL_MAX = 255


def quantize_layer(
    weights: torch.Tensor,
    biases: torch.Tensor,
    input_scale: float,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Quantize weights to int16 and biases to int32 accumulator units.

    ``input_scale`` maps input integer codes to real values (1/255 for pixels,
    1 for binary spikes). The returned accumulator scale is
    ``input_scale * weight_scale``.
    """
    weights_np = weights.detach().cpu().numpy().astype(np.float64, copy=False)
    biases_np = biases.detach().cpu().numpy().astype(np.float64, copy=False)
    if not np.isfinite(weights_np).all() or not np.isfinite(biases_np).all():
        raise ValueError("weights and biases must contain only finite values")
    if not np.isfinite(input_scale) or input_scale <= 0.0:
        raise ValueError("input_scale must be finite and positive")

    maximum = float(np.max(np.abs(weights_np)))
    if maximum == 0.0:
        raise ValueError("cannot derive a per-layer scale from all-zero weights")
    weight_scale = maximum / INT16_MAX
    quantized_weights = np.clip(
        np.rint(weights_np / weight_scale),
        INT16_MIN,
        INT16_MAX,
    ).astype(np.int16)

    accumulator_scale = input_scale * weight_scale
    quantized_biases_64 = np.rint(biases_np / accumulator_scale)
    int32_info = np.iinfo(BIAS_DTYPE)
    if np.any(quantized_biases_64 < int32_info.min) or np.any(
        quantized_biases_64 > int32_info.max
    ):
        raise OverflowError("quantized bias does not fit the selected int32 format")
    quantized_biases = quantized_biases_64.astype(BIAS_DTYPE)
    return quantized_weights, weight_scale, quantized_biases, accumulator_scale


def export_quantized_model(
    model: torch.nn.Module,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Export int16 weights, int32 accumulator-domain biases, and metadata."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    layers = (
        ("layer1", model.fc1, 1.0 / PIXEL_MAX),
        ("layer2", model.fc2, 1.0),
    )
    metadata: dict[str, Any] = {
        "format": "signed int16 weights; int32 accumulator-unit biases",
        "weight_dtype": "int16",
        "weight_signed_range": [INT16_MIN, INT16_MAX],
        "weight_quantization": (
            "q = clip(round_ties_to_even(w / (max_abs(w) / 32767)), "
            "-32768, 32767)"
        ),
        "input_representation": {
            "pixels": "uint8-range codes in [0, 255], widened to int64 for arithmetic",
            "pixel_scale": 1.0 / PIXEL_MAX,
            "hidden_spikes": "binary codes in [0, 1], widened to int64 for arithmetic",
            "hidden_spike_scale": 1.0,
        },
        "bias_representation": (
            "int32 bias codes in the corresponding layer accumulator units; "
            "bias_scale = input_scale * weight_scale"
        ),
        "network": {
            "input_features": model.fc1.in_features,
            "hidden_size": model.hidden_size,
            "output_classes": model.fc2.out_features,
            "num_steps": model.num_steps,
            "beta": model.beta,
            "threshold": model.threshold,
            "weight_layout": "PyTorch Linear [output, input], row-major",
        },
        "layers": {},
    }

    for name, layer, input_scale in layers:
        weights, weight_scale, biases, bias_scale = quantize_layer(
            layer.weight,
            layer.bias,
            input_scale,
        )
        if weights.dtype != np.int16:
            raise TypeError(f"{name} weights must be stored as int16")
        if weights.size == 0 or weights.min() < INT16_MIN or weights.max() > INT16_MAX:
            raise ValueError(f"{name} weights violate the signed-int16 range")
        np.save(destination / f"{name}_weights_int16.npy", weights)
        np.save(destination / f"{name}_scale.npy", np.array(weight_scale, dtype=np.float64))
        np.save(destination / f"{name}_bias_int32.npy", biases)
        np.save(destination / f"{name}_bias_scale.npy", np.array(bias_scale, dtype=np.float64))

        reconstructed = weights.astype(np.float64) * weight_scale
        flat_float = layer.weight.detach().cpu().numpy().reshape(-1)
        flat_quantized = weights.reshape(-1)
        flat_reconstructed = reconstructed.reshape(-1)
        samples = []
        for index in range(min(5, flat_float.size)):
            samples.append(
                {
                    "float": float(flat_float[index]),
                    "int16": int(flat_quantized[index]),
                    "reconstructed": float(flat_reconstructed[index]),
                }
            )
        metadata["layers"][name] = {
            "weight_shape": list(weights.shape),
            "weight_dtype": str(weights.dtype),
            "weight_file": f"{name}_weights_int16.npy",
            "weight_scale_file": f"{name}_scale.npy",
            "weight_scale": weight_scale,
            "bias_shape": list(biases.shape),
            "bias_dtype": str(biases.dtype),
            "bias_file": f"{name}_bias_int32.npy",
            "bias_scale_file": f"{name}_bias_scale.npy",
            "bias_scale": bias_scale,
            "integer_weight_min": int(weights.min()),
            "integer_weight_max": int(weights.max()),
            "weight_sample_comparison": samples,
        }

    with (destination / "quantization_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    return metadata


def _round_ratio_ties_to_even(values: np.ndarray, denominator: int) -> np.ndarray:
    """Round signed integer ratios to nearest, with ties to even."""
    absolute = np.abs(values)
    quotient = absolute // denominator
    remainder = absolute % denominator
    increment = (2 * remainder > denominator) | (
        (2 * remainder == denominator) & ((quotient & 1) == 1)
    )
    rounded = quotient + increment.astype(np.int64)
    return np.where(values < 0, -rounded, rounded)


class Int16SNN:
    """Integer reference inference using stored int16 weights and int32 biases.

    Dense products are accumulated explicitly in int64. Membrane state and
    threshold are represented in each layer's accumulator units. Beta=0.95 is
    implemented as 19/20 with nearest/ties-to-even integer rounding.
    """

    def __init__(self, export_dir: str | Path) -> None:
        directory = Path(export_dir)
        with (directory / "quantization_metadata.json").open(
            encoding="utf-8"
        ) as file:
            self.metadata = json.load(file)
        if self.metadata.get("weight_dtype") != "int16":
            raise ValueError("quantization metadata must specify signed int16 weights")
        if self.metadata.get("weight_signed_range") != [INT16_MIN, INT16_MAX]:
            raise ValueError("quantization metadata has an invalid signed-int16 range")
        network = self.metadata["network"]
        if network["beta"] != 0.95:
            raise ValueError("integer reference currently requires beta=0.95")
        if network["threshold"] <= 0.0 or network["num_steps"] < 1:
            raise ValueError("invalid threshold or number of time steps")

        self.hidden_size = int(network["hidden_size"])
        self.num_steps = int(network["num_steps"])
        self.beta = float(network["beta"])
        self.threshold = float(network["threshold"])
        if self.hidden_size < 1:
            raise ValueError("hidden_size in quantization metadata must be positive")
        input_features = int(network["input_features"])
        output_classes = int(network["output_classes"])
        if input_features != 28 * 28 or output_classes != 10:
            raise ValueError("quantization metadata has unsupported network dimensions")

        input_representation = self.metadata["input_representation"]
        pixel_scale = float(input_representation["pixel_scale"])
        hidden_spike_scale = float(input_representation["hidden_spike_scale"])
        if not np.isclose(pixel_scale, 1.0 / PIXEL_MAX, rtol=1e-12, atol=0.0):
            raise ValueError("pixel scale must match the [0, 255] input codes")
        if hidden_spike_scale != 1.0:
            raise ValueError("hidden binary spikes must use unit input scale")
        self.weights1, self.biases1, self.scale1, self.bias_scale1 = self._load_layer(
            directory,
            "layer1",
            (self.hidden_size, input_features),
            pixel_scale,
            self.metadata["layers"]["layer1"],
        )
        self.weights2, self.biases2, self.scale2, self.bias_scale2 = self._load_layer(
            directory,
            "layer2",
            (output_classes, self.hidden_size),
            hidden_spike_scale,
            self.metadata["layers"]["layer2"],
        )
        self.threshold1 = int(np.rint(self.threshold / self.bias_scale1))
        self.threshold2 = int(np.rint(self.threshold / self.bias_scale2))

    @staticmethod
    def _load_layer(
        directory: Path,
        name: str,
        expected_weight_shape: tuple[int, int],
        input_scale: float,
        layer_metadata: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        weights = np.load(directory / f"{name}_weights_int16.npy")
        biases = np.load(directory / f"{name}_bias_int32.npy")
        scale_array = np.load(directory / f"{name}_scale.npy")
        bias_scale_array = np.load(directory / f"{name}_bias_scale.npy")
        if scale_array.shape != () or bias_scale_array.shape != ():
            raise ValueError(f"{name} scale files must contain scalar values")
        scale = float(scale_array.item())
        bias_scale = float(bias_scale_array.item())
        if weights.dtype != np.int16:
            raise TypeError(f"{name} weights file must contain int16 values")
        if weights.shape != expected_weight_shape:
            raise ValueError(
                f"{name} weight shape must be {expected_weight_shape}, "
                f"not {weights.shape}"
            )
        if weights.min() < INT16_MIN or weights.max() > INT16_MAX:
            raise ValueError(f"{name} weights are outside the int16 range")
        if biases.dtype != np.int32:
            raise TypeError(f"{name} bias file must contain int32 values")
        expected_bias_shape = (expected_weight_shape[0],)
        if biases.shape != expected_bias_shape:
            raise ValueError(
                f"{name} bias shape must be {expected_bias_shape}, not {biases.shape}"
            )
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"{name} weight scale must be finite and positive")
        if not np.isfinite(bias_scale) or bias_scale <= 0.0:
            raise ValueError(f"{name} bias scale must be finite and positive")

        expected_bias_scale = input_scale * scale
        if not np.isclose(bias_scale, expected_bias_scale, rtol=1e-12, atol=0.0):
            raise ValueError(
                f"{name} bias scale does not equal input_scale * weight_scale"
            )
        if layer_metadata["weight_shape"] != list(weights.shape):
            raise ValueError(f"{name} weight dimensions disagree with metadata")
        if layer_metadata["bias_shape"] != list(biases.shape):
            raise ValueError(f"{name} bias dimensions disagree with metadata")
        if layer_metadata["weight_dtype"] != "int16":
            raise TypeError(f"{name} weight metadata must specify int16")
        if layer_metadata["bias_dtype"] != "int32":
            raise TypeError(f"{name} bias metadata must specify int32")
        if not np.isclose(
            float(layer_metadata["weight_scale"]), scale, rtol=1e-12, atol=0.0
        ):
            raise ValueError(f"{name} weight scale file disagrees with metadata")
        if not np.isclose(
            float(layer_metadata["bias_scale"]), bias_scale, rtol=1e-12, atol=0.0
        ):
            raise ValueError(f"{name} bias scale file disagrees with metadata")
        return weights, biases.astype(np.int64), scale, bias_scale

    @staticmethod
    def _lif_step(
        membrane: np.ndarray,
        current: np.ndarray,
        threshold: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        previous_reset = (membrane > threshold).astype(np.int64)
        decayed = _round_ratio_ties_to_even(membrane * 19, 20)
        pre_spike = decayed + current - previous_reset * threshold
        spikes = (pre_spike > threshold).astype(np.uint8)
        membrane = pre_spike - (spikes.astype(np.int64) - previous_reset) * threshold
        return membrane, spikes

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Return uint8 output spikes shaped (time, batch, 10)."""
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
            raise ValueError("images must have shape (batch, 1, 28, 28)")
        if images.shape[0] == 0:
            raise ValueError("images batch must not be empty")
        image_values = images.detach().cpu().numpy()
        if not np.isfinite(image_values).all():
            raise ValueError("images must contain only finite pixel values")
        if image_values.min() < 0.0 or image_values.max() > 1.0:
            raise ValueError("images must be normalized to [0, 1]")
        pixels = np.rint(image_values.reshape(images.shape[0], -1) * PIXEL_MAX).astype(
            np.int64
        )
        batch_size = pixels.shape[0]
        membrane1 = np.zeros((batch_size, self.hidden_size), dtype=np.int64)
        membrane2 = np.zeros((batch_size, 10), dtype=np.int64)
        output = np.zeros((self.num_steps, batch_size, 10), dtype=np.uint8)

        for step in range(self.num_steps):
            current1 = pixels @ self.weights1.astype(np.int64).T + self.biases1
            membrane1, hidden_spikes = self._lif_step(
                membrane1, current1, self.threshold1
            )
            current2 = (
                hidden_spikes.astype(np.int64)
                @ self.weights2.astype(np.int64).T
                + self.biases2
            )
            membrane2, output[step] = self._lif_step(
                membrane2, current2, self.threshold2
            )
        return torch.from_numpy(output)


def quantized_spike_counts(output_spikes: torch.Tensor) -> torch.Tensor:
    """Count binary integer-model spikes, returning int64 counts [batch, 10]."""
    if output_spikes.ndim != 3 or output_spikes.shape[-1] != 10:
        raise ValueError("output_spikes must have shape (time, batch, 10)")
    if output_spikes.shape[0] == 0 or output_spikes.shape[1] == 0:
        raise ValueError("output_spikes must have non-empty time and batch dimensions")
    return output_spikes.to(torch.int64).sum(dim=0)
