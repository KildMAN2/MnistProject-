"""Configurable static-current, fully connected LIF SNN for MNIST."""

from __future__ import annotations

from typing import TypeAlias

import torch
import snntorch as snn
from snntorch import surrogate
from torch import nn

INPUT_SIZE = 28 * 28
NUM_CLASSES = 10
DEFAULT_HIDDEN_SIZE = 256
DEFAULT_BETA = 0.95
DEFAULT_THRESHOLD = 1.0
DEFAULT_NUM_STEPS = 25

MembraneRecord: TypeAlias = dict[str, torch.Tensor]
ForwardResult: TypeAlias = torch.Tensor | tuple[torch.Tensor, MembraneRecord]


def encode_images(
    images: torch.Tensor, num_steps: int = DEFAULT_NUM_STEPS
) -> torch.Tensor:
    """Repeat static normalized pixel currents as (time, batch, 784)."""
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
        raise ValueError("images must have shape (batch, 1, 28, 28)")
    if num_steps < 1:
        raise ValueError("num_steps must be a positive integer")
    flattened = images.flatten(start_dim=1)
    return flattened.unsqueeze(0).expand(num_steps, -1, -1)


class MNISTSNN(nn.Module):
    """Two fully connected layers with LIF neurons at each simulation step.

    Normalized MNIST images shaped (batch, 1, 28, 28) are presented as the same
    pixel currents at every step. Forward returns output spikes shaped
    (time, batch, 10). Set ``record_membrane=True`` to also receive membrane
    histories.
    """

    def __init__(
        self,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        beta: float = DEFAULT_BETA,
        threshold: float = DEFAULT_THRESHOLD,
        num_steps: int = DEFAULT_NUM_STEPS,
    ) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be between 0 and 1")
        if threshold <= 0.0:
            raise ValueError("threshold must be positive")
        if num_steps < 1:
            raise ValueError("num_steps must be a positive integer")

        self.hidden_size = hidden_size
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.num_steps = num_steps

        spike_gradient = surrogate.fast_sigmoid(slope=25)
        self.fc1 = nn.Linear(INPUT_SIZE, hidden_size)
        self.lif1 = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_gradient,
            reset_mechanism="subtract",
            reset_delay=False,
        )
        self.fc2 = nn.Linear(hidden_size, NUM_CLASSES)
        self.lif2 = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_gradient,
            reset_mechanism="subtract",
            reset_delay=False,
        )

    def forward(
        self,
        images: torch.Tensor,
        record_membrane: bool = False,
    ) -> ForwardResult:
        """Simulate all time steps, optionally returning both membrane traces."""
        expected_shape = (1, 28, 28)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected_shape:
            raise ValueError("images must have shape (batch, 1, 28, 28)")
        if images.shape[0] == 0:
            raise ValueError("images batch must not be empty")

        input_currents = encode_images(images, self.num_steps)
        batch_size = images.shape[0]
        membrane_hidden = images.new_zeros((batch_size, self.hidden_size))
        membrane_output = images.new_zeros((batch_size, NUM_CLASSES))
        output_spikes: list[torch.Tensor] = []
        hidden_membranes: list[torch.Tensor] = []
        output_membranes: list[torch.Tensor] = []

        for step in range(self.num_steps):
            current_hidden = self.fc1(input_currents[step])
            spikes_hidden, membrane_hidden = self.lif1(
                current_hidden, membrane_hidden
            )

            current_output = self.fc2(spikes_hidden)
            spikes_output, membrane_output = self.lif2(
                current_output, membrane_output
            )
            output_spikes.append(spikes_output)

            if record_membrane:
                hidden_membranes.append(membrane_hidden)
                output_membranes.append(membrane_output)

        spike_record = torch.stack(output_spikes)
        if not record_membrane:
            return spike_record

        membrane_record = {
            "hidden": torch.stack(hidden_membranes),
            "output": torch.stack(output_membranes),
        }
        return spike_record, membrane_record


def count_output_spikes(output_spikes: torch.Tensor) -> torch.Tensor:
    """Sum output spikes over time, returning counts shaped (batch, 10)."""
    if output_spikes.ndim != 3 or output_spikes.shape[-1] != NUM_CLASSES:
        raise ValueError("output_spikes must have shape (time, batch, 10)")
    if output_spikes.shape[0] == 0 or output_spikes.shape[1] == 0:
        raise ValueError("output_spikes must have non-empty time and batch dimensions")
    return output_spikes.sum(dim=0)


def predict_from_spikes(output_spikes: torch.Tensor) -> torch.Tensor:
    """Return the most active class from a (time, batch, 10) spike record."""
    return count_output_spikes(output_spikes).argmax(dim=1)
