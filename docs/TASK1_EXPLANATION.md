# Task 1 â€” Spiking Neural Network for MNIST

## 1. Goal of Task 1

Task 1 builds a Python reference model that classifies handwritten digits from
MNIST. It uses PyTorch and snnTorch, a fully connected network, Leaky
Integrate-and-Fire (LIF) neurons, and surrogate-gradient training.

The model is deliberately small and explicit so that its inference behavior
can later be reproduced and checked in SystemVerilog. The current input method
is deterministic static-current presentation: the same pixel values are
presented at every simulation step. SystemVerilog is not implemented.
Training checkpoints store floating-point PyTorch parameters;
`quantization.py` exports signed-int16 layer weights and provides a separate
integer inference reference. The remaining hardware work is the SystemVerilog
implementation and its verification against this quantized Python reference.

Project files:

- `software/snn/dataset.py`: separate torchvision MNIST training and test datasets and
  their DataLoaders.
- `software/snn/model.py`: configurable SNN, static current presentation, LIF simulation,
  optional membrane recording, spike counting, and prediction.
- `software/snn/lif_demo.py`: five-step numerical example using the same snnTorch LIF
  configuration as the network.
- `software/snn/train.py`: training, validation-based checkpoint selection, and test
  evaluation after training.
- `software/snn/evaluate.py`: full test-set inference with example predictions and spike
  count debugging output.
- `software/quantization/quantization.py`: per-layer signed-int16 weight quantization, int32
  accumulator-domain bias export, metadata, and explicit integer-only
  inference.
- `software/quantization/quantize_and_compare.py`: export quantized layer files and compare
  floating-point and int16-weight inference accuracy over the test set.
- `artifacts/int16/`: generated NumPy arrays and JSON dimensions/scales for
  hardware use.
- `requirements.txt`: Python package dependencies.
- `docs/TASK1_EXPLANATION.md`: this technical explanation.

## 2. What MNIST Is

MNIST is a dataset of handwritten decimal digits. The official training split
contains 60,000 images, and the official test split contains 10,000 images.
Each image is labeled with one integer from 0 to 9.

Each image is a 28-by-28, single-channel grayscale picture. A pixel represents
the darkness or brightness at one location. `torchvision.datasets.MNIST`
provides separate training (`train=True`) and test (`train=False`) datasets.

## 3. Input Representation

The shared `torchvision.transforms.ToTensor()` transform converts each
8-bit pixel (0 through 255) to a floating-point value in [0, 1], approximately
by dividing by 255. Zero represents black and one represents maximum pixel
intensity. We do not apply mean/std normalization; keeping brightness in this
range gives a direct and consistent current amplitude.

One transformed image has shape:

```text
[1, 28, 28]        # channel, height, width
```

A mini-batch of `B` images and their labels has shapes:

```text
images: [B, 1, 28, 28]
labels: [B]
```

The image is flattened into 784 features because `28 * 28 = 784`. To supply
the SNN with a sequence, `model.py` repeats this same vector for `T` steps:

```text
[B, 1, 28, 28]
-> flatten -> [B, 784]
-> repeat over time -> [T, B, 784]
```

The static input value at time `t` is:

```text
input_current[b, t, i] = image[b, i]
```

This is a repeated multi-valued current input, not a binary input spike train.
The hidden and output LIF layers still generate binary spikes. Repetition is
deterministic: the same image and model parameters produce the same input
sequence.

An alternative is rate coding: treat pixel intensity `x_i` as the probability
of a binary input event, for example
`input_spike_i(t) ~ Bernoulli(x_i)`. Brighter pixels then produce more spikes
on average, but a random generator or a specifically defined deterministic
spike encoder is required. Different sampled input trains can produce
different outputs for the same image. We selected repeated static current for
this first hardware-oriented version because it is simpler to reproduce and
compare exactly in SystemVerilog.

## 4. Neural Networks: Neurons, Weights and Biases

A neural network contains connected computational units called neurons. In
this SNN, a linear layer combines its input values to calculate current for
neurons in the next LIF layer.

A weight determines how strongly one input contributes to one neuron. A bias
is a learned offset added to the weighted sum. Training updates these
parameters to make the output more consistent with the correct digit labels.
Each neuron has its own weights and bias so different neurons can learn to
respond to different input patterns.

## 5. Linear Layers

For input feature `i` and neuron `j`, a fully connected linear layer computes:

```text
y_j = sum_i (x_i * w_(j,i)) + b_j
```

Here, `x_i` is an input value, `w_(j,i)` is the weight connecting that input
to neuron `j`, `b_j` is that neuron's bias, and `y_j` is the current delivered
to the following LIF neuron.

For a batch, PyTorch's `nn.Linear` calculates `y = x @ W.T + b`:

- `Linear(784, hidden_size)` has weight shape
  `[hidden_size, 784]`, bias shape `[hidden_size]`, and output shape
  `[B, hidden_size]` at one time step.
- `Linear(hidden_size, 10)` has weight shape `[10, hidden_size]`, bias shape
  `[10]`, and output shape `[B, 10]` at one time step.

Weights and biases are floating point during gradient-based training.

## 6. What a Spiking Neural Network Is

A Spiking Neural Network (SNN) represents neuron outputs as discrete events
called spikes. Unlike a basic feed-forward ANN that computes its activations
in one static pass, an SNN updates internal neuron state across discrete time
steps. Whether a neuron fires now can depend on current input and membrane
potential accumulated from earlier steps.

In this project the pixels are repeated as static input currents, while the
network's LIF neurons produce spikes over time. Class evidence is represented
by output spike counts.

## 7. LIF Neuron

Both network layers use `snnTorch.Leaky` neurons with:

```text
beta = 0.95
threshold = 1.0
reset mechanism = subtract
reset_delay = False
```

The neuron receives an input current from its preceding linear layer. It
retains a fraction of its previous membrane potential, integrates the current,
fires if the membrane crosses its threshold, and subtracts threshold after a
spike.

## 8. Membrane Potential

Membrane potential is the state that records a neuron's accumulated input. In
the simplest form of the selected subtract-reset LIF update:

```text
pre_t = beta * membrane_(t-1) + current_t
spike_t = 1 if pre_t > threshold else 0
membrane_t = pre_t - spike_t * threshold
```

For snnTorch's immediate reset (`reset_delay=False`), it tracks a reset
indicator from the stored previous state. Its exact bookkeeping is:

```text
previous_reset_t = 1 if membrane_(t-1) > threshold else 0
pre_t = beta * membrane_(t-1) + current_t
       - previous_reset_t * threshold
spike_t = 1 if pre_t > threshold else 0
membrane_t = pre_t - (spike_t - previous_reset_t) * threshold
```

This correction matters for large overshoots that leave a stored potential
above threshold. When no such overshoot occurs, this reduces to the usual
equation above. Membrane state is stored from one step to the next within an
image simulation and is initialized to zero at the beginning of each
`MNISTSNN.forward` call. Therefore state does not carry from one batch/image
call into the next.

The `beta` parameter controls leak/decay:

- `beta = 1` retains all previous potential before adding current.
- Smaller beta values discard more of the prior state each step.
- Here `beta = 0.95`, so 95% of the previous membrane is retained per step.

## 9. Threshold and Spike Generation

The configured firing threshold is 1.0. snnTorch's forward spike is generated
when the pre-reset membrane is strictly greater than threshold:

```text
spike_t = H(pre_t - threshold)
```

`H` is the step function: it returns 1 for a positive argument and 0
otherwise. Thus a value exactly equal to 1.0 does not fire. A spike is a
binary event in the forward pass. With subtractive reset, one threshold is
subtracted when the neuron fires, leaving any residual membrane potential to
carry forward.

Numerical example (`beta=0.95`, threshold 1.0, initial membrane 0):

| t | Current | Previous membrane | Decayed membrane | Before threshold/reset | Spike | After reset |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6000 | 0.000000 | 0.000000 | 0.600000 | 0 | 0.600000 |
| 2 | 0.6000 | 0.600000 | 0.570000 | 1.170000 | 1 | 0.170000 |
| 3 | 0.0000 | 0.170000 | 0.161500 | 0.161500 | 0 | 0.161500 |
| 4 | 0.4000 | 0.161500 | 0.153425 | 0.553425 | 0 | 0.553425 |
| 5 | 0.6000 | 0.553425 | 0.525754 | 1.125754 | 1 | 0.125754 |

At steps 2 and 5, input plus retained membrane exceeds the threshold, so a
spike occurs and 1.0 is subtracted. `py -m software.snn.lif_demo` runs this example through
the actual snnTorch LIF implementation and checks the state updates.

## 10. Time Steps

The image is static, but the LIF state evolves. A neuron needs multiple steps
to integrate input over time, decay prior potential, produce spikes, and leave
a spike-count record for classification. With one step there is little
temporal integration and no meaningful multi-step count.

The default simulation uses `num_steps = 25`. Each step processes the same
flattened image current while the hidden and output membrane potentials
continue from the previous step. The membrane must be stored between steps;
otherwise each step would be independent and accumulated temporal behavior
would be lost.

## 11. Complete Network Architecture

The configurable model is `MNISTSNN` in `model.py`. Default parameters are:

| Parameter | Default |
|---|---:|
| Input features | 784 |
| Hidden neurons | 256 |
| Output neurons/classes | 10 |
| `beta` | 0.95 |
| `threshold` | 1.0 |
| `num_steps` | 25 |
| Reset | subtract threshold, immediate (`reset_delay=False`) |
| Input encoding | repeat static pixel current |
| Surrogate gradient | snnTorch fast sigmoid, slope 25 |

The hidden size, beta, threshold, and time-step count can be configured when
constructing the model or via `train.py` options.

```text
MNIST [B, 1, 28, 28]
        |
        v
flatten [B, 784] and repeat static values
        |
        v
input currents [T, B, 784]
        |
        v
Linear(784, 256) -> [B, 256] currents at each step
        |
        v
256 hidden LIF neurons -> [T, B, 256] spikes
        |
        v
Linear(256, 10) -> [B, 10] currents at each step
        |
        v
10 output LIF neurons -> [T, B, 10] output spikes
        |
        v
sum over time [B, 10] -> argmax -> [B] predicted digits
```

## 12. Forward Pass

For every call, `MNISTSNN.forward(images)`:

1. Validates the input shape `[B, 1, 28, 28]`.
2. Flattens and repeats the pixels to `[T, B, 784]`.
3. Creates fresh zero membrane tensors `[B, hidden_size]` and `[B, 10]`.
4. For each of `T` steps, computes the first linear current and hidden LIF
   spikes, then computes the second linear current and output LIF spikes.
5. Records output spikes at every step and stacks them as `[T, B, 10]`.

With `record_membrane=True`, the forward pass additionally returns membrane
histories for both layers:

```text
hidden membrane history: [T, B, hidden_size]
output membrane history: [T, B, 10]
```

## 13. Training

Training adjusts the floating-point linear weights and biases so predicted
digits agree with their labels.

```text
image + label
-> forward SNN across T steps
-> output spikes
-> spike counts / class scores
-> classification loss
-> backward through time
-> gradients
-> Adam update
-> next batch; repeat for each epoch
```

`train.py` uses the official MNIST training dataset and reserves 5,000 of its
60,000 examples for validation. The remaining 55,000 examples are shuffled
for updates. The separate official test set is not used to select checkpoints.
Defaults are 5 epochs, learning rate 0.001, and batch size 128. These and
`num_steps` are configurable using `--epochs`, `--learning-rate`,
`--batch-size`, and `--num-steps`. CUDA is used if available; otherwise
training runs on CPU.

Each batch is passed through the model. Cross-entropy loss is calculated from
the 10 output spike counts per sample and the integer ground-truth labels.
Adam updates weights after every batch. Training accuracy is measured from the
batch predictions; epoch loss is averaged across samples. Validation accuracy
selects the best checkpoint. The checkpoint saves the state dictionary,
architecture parameters, epoch, training metrics, and validation accuracy.

## 14. Surrogate Gradients

A hard spike is a thresholded step function. Its derivative is zero almost
everywhere and undefined at the threshold, which prevents ordinary
gradient-based learning from propagating useful gradients through the spike
operation.

The model uses `snnTorch`'s fast-sigmoid surrogate gradient with slope 25.
Forward evaluation remains threshold-based, but during the backward pass a
smooth approximate derivative is used around the threshold. PyTorch autograd
can then calculate gradients for the linear-layer weights and biases.

Because membrane state and spikes are connected across simulation steps,
gradients propagate through the time sequence. This is Backpropagation
Through Time (BPTT). PyTorch and snnTorch perform this training computation;
the later hardware inference implementation will use fixed trained
parameters and will not need surrogate gradients or BPTT.

## 15. Spike Counting

At each step, the output LIF layer produces a vector of 10 spike values for
each sample. `count_output_spikes` sums across the time dimension:

```text
spike_count[b][j] = sum(t=0..T-1, output_spikes[t][b][j])
```

Input spike record shape is `[T, B, 10]`; the count result is `[B, 10]`.
Each count is from zero through `T`. Counting represents total class-neuron
activity over the simulation rather than activity at only one step.

## 16. Classification

Output neuron index `j` represents MNIST digit `j`: output 0 means digit 0,
and output 9 means digit 9. One output spike is one threshold-triggered event
from that digit's output neuron at one simulation step.

The prediction is the index of the largest count:

```text
prediction[b] = argmax(j=0..9, spike_count[b][j])
```

Example for one image across 25 steps:

| Neuron / digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Spike count | 1 | 0 | 3 | 2 | 1 | 0 | 4 | 8 | 5 | 1 |

The largest value is 8 at index 7, so argmax returns 7 and the predicted
digit is **7**. In a tie, PyTorch's `argmax` returns the first/lowest index
having the maximum count; a matching hardware implementation should retain
the earlier index unless a strictly larger count is found.

`evaluate.py --examples 5` prints the true digit, predicted digit, and all ten
counts for five examples. Hardware can implement ten counters, incrementing
each by its output spike bit each step, then compare the counters after step
`T-1`. At `T=25`, an unsigned 5-bit counter can represent 0 through 25.

## 17. Evaluation and Accuracy

Training updates weights; inference uses fixed checkpoint weights and does
not calculate gradients or update parameters. The training split is used to
fit parameters. The validation subset chooses the best checkpoint. The
official test set consists of unseen examples and measures generalization.

Testing on unseen examples matters because high training accuracy alone may
mean the network memorized its examples. Overfitting occurs when training
performance is high but performance on new data is worse. The test split is
kept separate from training and checkpoint selection.

For each test image, evaluation initializes fresh membrane states by calling
the model, simulates its saved `num_steps`, counts output spikes, and selects
the argmax digit. Accuracy is:

```text
accuracy_percent = 100 * correct_predictions / total_test_samples
```

`evaluate.py` evaluates the floating-point training checkpoint on the entire
official test split inside `torch.no_grad()`. Its reported accuracy is
explicitly labeled floating-point; use `quantize_and_compare.py` to evaluate
the signed-int16 inference path.
By default, it prints five sample predictions and all ten counts for each,
then prints the number correct, total samples, and final accuracy percentage.
Use `--examples N` to change the number displayed or `--examples 0` to
suppress sample lines.

The completed five-epoch static-current training run selected epoch 4 as the
best validation checkpoint (validation accuracy **96.98%**). Evaluating that
checkpoint on the complete official test set reported **97.13% accuracy
(9,713 / 10,000)**. The `evaluate.py` run was repeated independently and
returned the same correct count, total, and percentage. The prior **96.65%
(9,665 / 10,000)** result belongs to the previous stochastic rate-coded model
and checkpoint `artifacts/checkpoints/mnist_snn.pt`, not the current static-current architecture.

To reproduce evaluation:

```powershell
py -m software.snn.evaluate --checkpoint artifacts/checkpoints/mnist_snn_static.pt
```

The best checkpoint is selected using validation accuracy; the official test
set is evaluated only after restoring the selected checkpoint.

## 18. Final Trained Parameters

For the default hidden size 256, the model's parameter counts are:

### Weights

```text
first layer:  784 * 256 = 200,704
second layer: 256 * 10  =   2,560
total weights:           = 203,264
```

### Biases

```text
first layer:  256
second layer:  10
total biases: 266
```

There are 203,530 trainable floating-point parameters in total. The weight
matrix shapes are `[256, 784]` and `[10, 256]`; bias shapes are `[256]` and
`[10]`. Counts depend on `hidden_size`: weights are
`784 * hidden_size + 10 * hidden_size`, biases are `hidden_size + 10`.

The best-selected floating-point values are saved in
`artifacts/checkpoints/mnist_snn_static.pt` under `model_state_dict`, along with `hidden_size`,
`beta`, `threshold`, `num_steps`, epoch, training metrics, and validation
accuracy. In the verified run the checkpoint came from epoch 4, with
validation accuracy 96.98%; its parameters are float32 PyTorch tensors. The
test accuracy of that checkpoint is 97.13% (9,713 / 10,000). The earlier
`artifacts/checkpoints/mnist_snn.pt` is a rate-coded checkpoint and should not be used as if it were
trained with the present static-current input method.

`quantize_and_compare.py` exports quantized weights, scale factors, biases,
and dimensions to `artifacts/int16/`, then compares floating-point and explicit
integer inference on the full test split. The saved float checkpoint remains
the training artifact; the exported int16 weights and integer inference are
the hardware-oriented inference artifacts.

The verified int16-weight comparison on the same 10,000-example test split
reported **97.13% floating-point accuracy**, **97.13% int16-weight accuracy**,
and **0.00 percentage-point accuracy loss** at two-decimal display precision.
The measured correct counts were 9,713/10,000 for each inference path. This
is a measured result for these exact checkpoint parameters, scales, and
integer arithmetic rules, not a guarantee for future training runs or
hardware implementations.

## 19. What Will Be Needed for the Future SystemVerilog Implementation

The SystemVerilog design should reproduce inference, not Python training.
It will need to:

1. Load and represent the 784 pixel inputs and the two trained linear layers.
2. Use the same input normalization and repeat each static pixel current over
   exactly `T` steps.
3. Implement `Linear(784, hidden_size)` and `Linear(hidden_size, 10)` weighted
   sums with defined weight layout, bias addition, and accumulator width.
4. Store one membrane value per hidden and output neuron between time steps,
   initialize state for a new sample, apply beta/leak and the exact threshold
   comparison, emit spikes, and apply the subtractive immediate reset.
5. Count each of the 10 output spikes for all time steps and implement
   argmax with the same lowest-index tie rule as PyTorch.
6. Load the signed-int16 layer weights and scales exported under
   `artifacts/int16/`; use the int32 accumulator-domain biases and their scales
   from the same directory.
7. Match the explicit integer reference's pixel coding, integer accumulators,
   beta rounding, threshold quantization, comparison and reset rules.
8. Compare intermediate currents, membrane traces, spike records, counts, and
   predicted digits against the Python integer-inference reference using fixed
   test vectors.

The hardware implementation will also need explicit decisions for memory
organization, cycle schedule, reset/enable behavior, input/output interface,
overflow and saturation behavior, and whether multiplication or another
optimized arithmetic mapping is appropriate. These are future implementation and verification tasks. This project provides
the floating-point training/reference model and a Python integer inference
reference, but it does not yet include SystemVerilog.

# 16-Bit Integer Weight Quantization

## Why training uses floating-point weights

PyTorch gradient-based training relies on differentiable floating-point
parameters. The backward pass computes small gradients and Adam updates these
parameters repeatedly. The trained checkpoint `artifacts/checkpoints/mnist_snn_static.pt` therefore
stores float32 shadow/training weights. These are not the final hardware
weights.

The final inference weights for the hardware-oriented path are signed 16-bit
integers (`int16`). A signed 16-bit value ranges from **-32768 to +32767**.
Integer weights have a fixed storage width and can be represented as signed
two's-complement values in hardware.

## Per-layer scale, rounding, clipping, and reconstruction

Weights in different Linear layers can have different numeric ranges, so each
layer receives its own scale. For a floating-point layer weight tensor `W`,
the implementation uses:

```text
scale = max(abs(W)) / 32767
q = round(W / scale)
q_int16 = clip(q, -32768, 32767)
```

`round` is nearest with NumPy's ties-to-even rule. Clipping guarantees every
stored value fits signed int16. The scale is saved independently for each
layer. A dequantized approximation can be reconstructed as:

```text
W_approx = q_int16 * scale
```

In code, `quantization.py` implements per-layer conversion, verifies integer
type and range, and exports scale metadata. `artifacts/int16/layer1_weights_int16.npy`
and `artifacts/int16/layer2_weights_int16.npy` hold the actual int16 arrays, not
float-cast weights.

### Numerical example

Suppose one layer has `max(abs(W)) = 0.20` and one weight is `W_i = -0.075`:

```text
scale = 0.20 / 32767
      â‰ˆ 0.0000061037

q_i = round(-0.075 / 0.0000061037)
    = round(-12287.625...)
    = -12288

W_approx_i = -12288 * 0.0000061037
           â‰ˆ -0.0750023
```

The small difference `W_approx_i - W_i` is quantization error. With this
max-based scale the largest-magnitude layer weight maps near 32767. Different
layers have separate scales because their maxima can differ. The exact
quantized values, reconstructed samples, scale, and min/max are printed by
the export command and recorded in its metadata JSON.

## Bias representation recommendation

Leaving biases floating point would make the supposed integer reference
depend on floating-point arithmetic and would not provide a complete
hardware-ready linear layer. Quantizing the bias to int16 with the weight
scale alone is also incorrect because the bias is added after multiplication
and is measured in layer-current units.

This project therefore represents each bias as **signed int32 in the same
integer accumulator units as its layer's dot product**. This is a practical
hardware-friendly choice with headroom beyond the weight storage width, while
keeping the linear operation entirely integer. For a layer whose input integer
code represents `input_scale` real units and whose weight scale is
`weight_scale`:

```text
accumulator_scale = input_scale * weight_scale
bias_int32 = round(bias_float / accumulator_scale)
real_accumulator_approx = integer_accumulator * accumulator_scale
```

For layer 1, image pixels are rounded to integer codes `p` in [0, 255], which
fit the uint8 input representation, so `input_scale = 1/255`. The layer-1
bias scale is therefore
`(1/255) * layer1_weight_scale`. For layer 2, the inputs are binary hidden
spikes coded as 0 or 1, so `input_scale = 1`, and its bias scale equals the
layer-2 weight scale. Biases are saved as `layer1_bias_int32.npy` and
`layer2_bias_int32.npy`; their accumulator scales are saved separately.
Quantization rejects a bias that does not fit int32 rather than silently
clipping it.

## Explicit integer inference path

`Int16SNN` in `quantization.py` is a separate inference implementation. It
loads int16 weight files and int32 biases, converts integer operands to int64
only to perform safe widened multiply/accumulate operations, and never copies
quantized weights into the original float `nn.Linear` layers. Pixels are
converted from normalized floats to integer codes in the uint8 range and then
widened to int64 for safe arithmetic; hidden outputs are binary integer
spikes.

Each layer's accumulator and membrane are in its own scale units. The LIF
threshold is quantized into those units. With the project's `beta=0.95`, the
integer reference computes leak as `round_ties_to_even(19 * membrane / 20)`,
then adds the integer current, performs a strict integer threshold comparison,
and applies the same immediate subtractive-reset bookkeeping. Division
rounding and threshold quantization are explicit and may differ slightly
from float arithmetic near a firing boundary. This makes the reference
deterministic and useful for hardware comparison.

## Accuracy effect and verification

Rounding changes each weight by at most approximately half a quantization
step before any clipping. Small errors can change a membrane value near
threshold, which can add or remove a spike and may alter the final class.
Therefore float accuracy cannot be assumed to equal int16-weight accuracy.

Run the end-to-end export and comparison:

```powershell
py -m software.quantization.quantize_and_compare --checkpoint artifacts/checkpoints/mnist_snn_static.pt
```

The script evaluates both the original float SNN and the separate integer
inference path over all 10,000 MNIST test samples and prints both accuracies,
their difference, and several paired predictions/counts. It also reports
each layer's int16 min/max, scale, dimensions, int32 bias scale, and several
float/int16/reconstructed weight examples. It checks every exported weight's
dtype and signed-int16 range.

For the currently trained checkpoint, the measured export details are:

| Layer | Weight shape | int16 min / max | Weight scale | Bias representation |
|---|---:|---:|---:|---|
| 1 | `[256, 784]` | `-32767 / 30834` | `6.7606623363e-6` | int32, scale `2.6512401319e-8` |
| 2 | `[10, 256]` | `-32767 / 26127` | `6.0719878476e-6` | int32, scale `6.0719878476e-6` |

### Final numeric-format audit and arithmetic verification

The final audit checked project source, documentation, and generated export
metadata for weight-format assumptions. No 8-bit or unsigned weights are
used. Both Linear-layer weight files are stored as signed `numpy.int16`;
uint8 is only the input-pixel and binary-spike code range, and floating-point
weights are retained for training and the separate floating-point comparison
path. Conversion to int16 occurs only after scaling, nearest/ties-to-even
rounding, and clipping.

The integer reference loads each layer's `*_weights_int16.npy`,
`*_scale.npy`, `*_bias_int32.npy`, and `*_bias_scale.npy` files directly. It
checks the stored types and dimensions, signed weight range, positive finite
scales, agreement between scale files and metadata, and the relationship
`bias_scale = input_scale * weight_scale`. It uses the exported weight codes
without converting them back to floating-point Linear weights.

For integer input codes `x_code`, stored signed weights `q`, and int32 bias
codes `b_code`, the integer layer operation is:

```text
accumulator = sum(x_code * q) + b_code
approximate_real_current = accumulator * input_scale * weight_scale
```

The second equation follows because the exported bias code is in the same
accumulator units, with `bias_scale = input_scale * weight_scale`. The Python
reference widens stored int16 weights and int32 biases to int64 for the
multiply/accumulate; that widening is arithmetic precision, not a change to
the exported weight format. Layer 1 uses integer pixel codes in [0, 255] and
`input_scale = 1/255`; layer 2 uses binary spike codes in [0, 1] and
`input_scale = 1`.

The full test-set comparison using the current checkpoint and files under
`artifacts/int16/` produced:

| Verification | Result |
|---|---:|
| Floating-point test accuracy | 97.13% (9,713/10,000) |
| Signed-int16-weight integer-reference accuracy | 97.13% (9,713/10,000) |
| Prediction agreement between paths | 100.00% |
| Accuracy difference at displayed precision | 0.00 percentage points |

This verifies the current export and implementation together. It does not
replace future bit-accurate hardware checks for accumulator width, rounding,
and overflow behavior.

## Exported files and SystemVerilog storage

The command writes these files to the selected export directory:

| File | Contents |
|---|---|
| `layer1_weights_int16.npy` | Layer 1 int16 weights, shape `[256, 784]` |
| `layer1_scale.npy` | Layer 1 floating scale used to interpret weights |
| `layer1_bias_int32.npy` | Layer 1 int32 bias in accumulator units |
| `layer1_bias_scale.npy` | Layer 1 real units represented by one bias code |
| `layer2_weights_int16.npy` | Layer 2 int16 weights, shape `[10, 256]` |
| `layer2_scale.npy` | Layer 2 floating scale used to interpret weights |
| `layer2_bias_int32.npy` | Layer 2 int32 bias in accumulator units |
| `layer2_bias_scale.npy` | Layer 2 real units represented by one bias code |
| `quantization_metadata.json` | Dimensions, LIF settings, layouts, scales, ranges, and examples |

Weights use PyTorch Linear row layout `[output, input]`; a hardware loader can
serialize these arrays in the selected ROM/RAM order. The JSON metadata
captures layer dimensions and scales so values can be reconstructed and the
hardware interface can be specified precisely.

## Wider products and accumulators

A signed 16-bit weight multiplied by another signed 16-bit value can require
a signed 32-bit product: a 16-by-16 multiplication is not generally representable
in 16 bits. A Linear neuron then adds many such products. Summing `N` terms
needs additional guard bits (approximately `ceil(log2(N))` beyond the product
width for a conservative bound), plus enough headroom for bias. For example,
a full 784-term sum of arbitrary signed 16-bit products needs substantially
more than 32 bits; this Python reference accumulates in int64 to avoid silent
overflow.

In this particular input pipeline layer 1 multiplies uint8-range pixel codes
(widened to int64 in the Python arithmetic) by int16 weights, while layer 2
multiplies binary spikes by int16 weights. These restrictions can permit
narrower hardware accumulators after worst-case bounds for inputs, biases, and
membrane state have been formally established. Until that bound analysis is
complete, the hardware should use a sufficiently wide accumulator and
explicitly define overflow/saturation behavior. The exported weight format is
int16; it does not imply that products, biases, or membrane state should also
be only 16 bits.
