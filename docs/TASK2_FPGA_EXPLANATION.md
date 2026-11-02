# Task 2 — Equivalent FPGA SNN Implementation

## Goal

Implement an FPGA/SystemVerilog model whose inference behavior is equivalent
to the INT16 Python reference implementation from Task 1.

Status: **not started.** This document currently records only the software
reference the hardware must match. No FPGA architecture has been designed and
no RTL exists yet (`fpga/` contains empty placeholder directories).

## Software Reference

The hardware must match `Int16SNN` in
[`software/quantization/quantization.py`](../software/quantization/quantization.py),
using the parameters in [`artifacts/int16/`](../artifacts/int16/). The
floating-point model in `software/snn/model.py` is the training model, not the
bit-exact reference.

### Network

| Item | Value |
|---|---|
| Architecture | 784 -> Linear -> LIF (hidden) -> Linear -> LIF (output) -> spike count -> argmax |
| Input size | 784 (28 x 28 MNIST image, flattened) |
| Hidden size | 256 |
| Output size | 10 (one neuron per digit) |
| Simulation time steps | 25 |
| beta | 0.95 (integer model: 19/20) |
| Threshold | 1.0 real units |
| Reset | subtractive, `reset_delay=False` (see below) |
| Layer 1 weight shape | `[256, 784]` (`[output, input]`, row-major) |
| Layer 2 weight shape | `[10, 256]` (`[output, input]`, row-major) |
| Weight format | signed int16, range -32768 ... 32767 |
| Bias format | signed int32, in each layer's accumulator units |
| Pixel representation | `round(pixel_float * 255)`, integer code 0 ... 255; layer-1 input scale 1/255 |
| Hidden spike representation | binary 0/1; layer-2 input scale 1 |
| Output spike representation | binary 0/1 per neuron per time step |

Input is **static current**: the same 784 pixel codes are applied at every
one of the 25 time steps. Membrane potentials start at 0 for each image.

### Quantization parameters (current checkpoint)

| | Layer 1 | Layer 2 |
|---|---|---|
| Weight scale | 6.76066233626e-06 | 6.07198784756e-06 |
| Bias scale (= input scale x weight scale) | 2.65124013187e-08 | 6.07198784756e-06 |
| Integer weight min / max | -32767 / 30834 | -32767 / 26127 |
| Integer threshold (accumulator units) | 37718198 | 164691 |

Weight quantization: `q = clip(round_ties_to_even(w / scale), -32768, 32767)`
with `scale = max(abs(w)) / 32767` per layer. Reconstruction: `w ~ q * scale`.
Integer threshold = `rint(1.0 / bias_scale)`.

### Per-time-step integer arithmetic

For each layer and each time step, with all arithmetic in wide integers
(the Python reference uses int64):

```text
current      = sum_i(input_code[i] * weight_int16[j][i]) + bias_int32[j]
previous_reset = (membrane > threshold)
decayed      = round_ties_to_even(19 * membrane / 20)
pre_spike    = decayed + current - previous_reset * threshold
spike        = (pre_spike > threshold)                 # strict comparison
membrane_new = pre_spike - (spike - previous_reset) * threshold
```

Layer 1 produces 256 hidden spikes; they are the 0/1 input codes to layer 2,
which produces the 10 output spikes. The exact rounding rule for the 19/20
decay (nearest, ties to even, symmetric for negative values) is implemented
by `_round_ratio_ties_to_even` in `quantization.py`.

### Classification

```text
spike_count[j] = sum over t = 0..24 of output_spike[t][j]
prediction     = argmax(spike_count)    # ties -> lowest index (as torch.argmax)
```

### Current reference result

On the full 10,000-image MNIST test set the integer reference scores
**97.13% (9,713 / 10,000)**, identical to the floating-point model, with 100%
prediction agreement.

## Hardware Equivalence Requirement

The SystemVerilog implementation must be verified against the **integer
Python reference model**, not only against the floating-point PyTorch model.

For the same input image, we eventually want to compare:

```text
Python INT16 reference
vs
SystemVerilog simulation
vs
FPGA implementation
```

including intermediate values where possible: pixel codes, layer-1
accumulators, hidden membranes and spikes, layer-2 accumulators, output
membranes, per-step output spikes, spike counts, and the predicted digit.

## Files Task 2 Will Consume

From `artifacts/int16/`:

- `layer1_weights_int16.npy`, `layer1_bias_int32.npy`,
  `layer1_scale.npy`, `layer1_bias_scale.npy`
- `layer2_weights_int16.npy`, `layer2_bias_int32.npy`,
  `layer2_scale.npy`, `layer2_bias_scale.npy`
- `quantization_metadata.json`

## Open Items (to resolve before RTL)

- Worst-case accumulator and membrane bit-width analysis (the Python
  reference uses int64; hardware needs proven bounds or defined saturation).
- Hardware implementation of the 19/20 ties-to-even decay.
- Memory-initialization (`.mem`) file format and weight ordering.
- Test-vector format for cross-checking intermediate values.
