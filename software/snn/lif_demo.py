"""Demonstrate the same immediate-reset LIF neurons used by the MNIST SNN."""

from __future__ import annotations

import torch
import snntorch as snn
from snntorch import surrogate

from software.snn.model import DEFAULT_BETA as BETA, DEFAULT_THRESHOLD as THRESHOLD


INPUT_CURRENTS = (0.6, 0.6, 0.0, 0.4, 0.6)


def main() -> None:
    neuron = snn.Leaky(
        beta=BETA,
        threshold=THRESHOLD,
        spike_grad=surrogate.fast_sigmoid(slope=25),
        reset_mechanism="subtract",
        reset_delay=False,
    )
    membrane: torch.Tensor | None = None
    previous = 0.0

    print(
        "t | current | previous | decayed | before threshold | spike | after reset"
    )
    for step, current in enumerate(INPUT_CURRENTS, start=1):
        decayed = BETA * previous
        previous_reset = int(previous > THRESHOLD)
        before_threshold = decayed + current - previous_reset * THRESHOLD

        spike_tensor, membrane = neuron(
            torch.tensor([current], dtype=torch.float32), membrane
        )
        spike = int(spike_tensor.item())
        expected_membrane = before_threshold - (spike - previous_reset) * THRESHOLD

        if not torch.isclose(
            membrane,
            torch.tensor([expected_membrane], dtype=membrane.dtype),
            atol=1e-6,
        ).all():
            raise RuntimeError("snnTorch LIF state did not match the documented update")

        print(
            f"{step} | {current:.4f} | {previous:.6f} | {decayed:.6f} | "
            f"{before_threshold:.6f} | {spike} | {float(membrane.item()):.6f}"
        )
        previous = float(membrane.item())


if __name__ == "__main__":
    main()
