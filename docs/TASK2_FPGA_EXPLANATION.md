# Task 2 — Equivalent FPGA SNN Implementation

## Goal

Implement an FPGA/SystemVerilog model whose inference behavior is equivalent
to the INT16 Python reference implementation from Task 1.

Status: **architecture designed, no RTL yet.** The first half of this document
records the software reference the hardware must match; the second half
(from "FPGA Architecture" onward) is the hardware design. `fpga/` still
contains only placeholder directories.

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

# FPGA Architecture

## Selected architecture

**One MAC, neuron-serial, with the layer-1 current cached per image.**

Key observations from the Python reference (`Int16SNN`):

1. Pixels and layer-1 bias are constant over the 25 time steps, so the layer-1
   current `cur1[j]` is identical every step. Computing it once per image and
   caching it is *exactly* equivalent (a scheduling change, not a numeric one).
   A `RECOMPUTE_L1` parameter can restore the literal per-step schedule.
2. Hidden spikes are 0/1, so layer 2 needs **no multiplier**:
   `acc += spike ? w : 0`.
3. Only layer 1 needs a multiplier: 9-bit signed pixel (0..255) x int16 weight.

### Datapath options

| | A: 1 MAC serial | B: P lanes along inputs (P=16) | C: A + cached cur1 (selected) | D: P_N neurons in parallel |
|---|---|---|---|---|
| DSP (L1 MAC only; decay adds 0 with the shift/add unit, 6 with the reciprocal multiplier) | 1 | 16 | 1 | P_N |
| Weight BRAM | 98 RAMB36 | 98 (wide 256-bit words) | 98 | 98, banked by neuron |
| L1 cycles/image | ~5.1 M (recompute each step) | ~320 k | ~201 k once | ~201 k / P_N |
| Weight bandwidth | 16 b/clk | 256 b/clk | 16 b/clk | 16·P_N b/clk |
| Control | simplest | adder tree + padding (784/16=49) | simple + cache RAM | banking, P_N-way fan-out |
| Area | smallest | largest DSP/LUT | small (+1 BRAM) | grows with P_N |

Option C is chosen: minimal area, trivially sequential weight addressing,
and ~16x faster than A. B/D are possible later speed-ups; they do not change
numerics because integer addition is associative.

## Module hierarchy

```
snn_top
├── snn_controller        FSM + counters (below)
├── pixel_buffer          784 x 8 RAM, written from outside
├── rom_sync (w1)         200,704 x 16, sequential address
├── rom_sync (b1)         256 x 32
├── l1_mac                9b signed x int16 -> 24b, accumulate to ACC1 (35b)
├── simple_dp_ram (cur1)  256 x 35
├── simple_dp_ram (mem1)  256 x 38
├── lif_core (L1)         prev_reset cmp, decay19_20, base add, 2 cond. thr subtracts, strict spike cmp
├── hidden_spike_reg      256 FFs, 8->256 write decode, 256:1 read mux
├── rom_sync (w2)         2,560 x 16
├── rom_sync (b2)         10 x 32
├── l2_acc                gated accumulator, ACC2 (33b generic)
├── lif_core (L2)         same RTL, different parameters
├── spike_counter x10     5 bits each (0..25)
└── argmax10              strict '>' keeps the lowest index
```

Top interface: pixel write port (`addr[9:0]`, `data[7:0]`, `we`), `start`,
`busy`, `done`, `pred[3:0]`, `counts[10][5]`.
`snn_params_pkg.sv` (generated) holds widths and integer thresholds.

# Datapath

## Pixel x weight: explicit signedness

Decision: the pixel is stored as `logic [7:0] pixel_u` (raw MNIST byte,
unsigned 0..255) and converted to a **signed 9-bit** operand by prepending a
zero: `logic signed [8:0] pixel_s = {1'b0, pixel_u};`. The weight is
`logic signed [15:0] weight_s` (two's complement int16, read from the `.mem`
file). The product is

```
logic signed [23:0] product_s;
product_s = pixel_s * weight_s;          // both operands signed, result 24 bit
```

**Range proof.** pixel in [0, 255], weight in [-32768, 32767]:

- minimum = 255 x (-32768) = -8,355,840
- maximum = 255 x 32767 = +8,355,585
- signed 24-bit range = [-8,388,608, +8,388,607], so every product fits
  (|min| < 2^23 = 8,388,608). 23 bits would be too small for the sign, so
  **24 signed bits is the minimum**. (Exported weights actually span
  [-32767, 30834], a subset.)

**Why `{1'b0, pixel_u}` and not a bare 8-bit operand.** In SystemVerilog a
multiplication is signed only if *all* operands are signed. If `pixel_u`
(unsigned) is mixed with a signed weight, the expression is evaluated as
unsigned: the weight is zero-extended, so a negative weight such as -1
(`16'hFFFF`) becomes +65535 and the product is wrong. With both operands
declared `signed`, the narrower operand is sign-extended to the context width
(24 bits for the assignment), so a negative weight is extended correctly. A
9-bit signed pixel with a leading 0 stays non-negative (0..255), so
sign-extension adds only zeros. Never use `$signed(pixel_u)` on the 8-bit
value: `8'd200` would become -56. The only safe forms are
`$signed({1'b0, pixel_u})` or the declared `pixel_s` above. Likewise the
accumulator must be declared `logic signed [34:0] acc` (generic-safe) and the
product is added as-is (`acc <= acc + product_s;` with `product_s` signed, so
it is sign-extended to 35 bits); bias is `logic signed [31:0]`.

Layer 2 has no multiplier: `acc2 <= acc2 + (spike ? weight_s : 16'sd0);` -
`weight_s` is signed, so it is sign-extended into the signed accumulator.

Final types: pixel `logic signed [8:0]` (from the unsigned 8-bit byte,
0..255), weight `logic signed [15:0]`, product `logic signed [23:0]`. Verified
numerically: the min/max products above were computed from the full
operand extremes, and the exported weight dtype is int16 with min/max
-32767/30834.

## Layer equations
Layer 1 (per image, per neuron j): `ACC1[j] = b1[j] + sum_i pixel[i]*W1[j][i]`
(784 MAC cycles + 3 pipeline cycles: BRAM read, product register,
accumulator register). Result stored in the `cur1` RAM.

Layer 2 (per time step): `ACC2[k] = b2[k] + sum_{j: s1[j]=1} W2[k][j]`
(256 gated adds per output neuron).

# Memory Organization

| Memory | Size | Mapping |
|---|---|---|
| L1 weight ROM | 200,704 x 16 = 3.21 Mbit | 98 RAMB36 (2K x 18) |
| L2 weight ROM | 2,560 x 16 | 2 RAMB36 |
| Bias ROMs | 256 x 32, 10 x 32 | small ROM / LUTRAM |
| Pixel buffer | 784 x 8 | 1 RAMB18 |
| `cur1` | 256 x 35 | 1 RAMB36 |
| `mem1` | 256 x 38 | 1 RAMB36 or LUTRAM |
| `mem2` | 10 x 36 | FFs |

Total about 100+ RAMB36. **An xc7a35t (50 BRAM36) is too small**; xc7a100t
(135) or Zynq-7020 (140) is suitable.

**Address formulas** (arrays are C-contiguous `[out][in]`, verified on random
samples):

- L1: `addr1 = j*784 + i`. In the neuron-serial order the address simply
  counts 0..200,703 (18 bits) - no multiplier.
- L2: `addr2 = k*256 + j` (12 bits, 0..2,559).

**`.npy` -> `.mem` flow** (`fpga/scripts`): load the `.npy`, flatten in C
order, write one two's-complement hex word per line (4 digits for weights,
8 digits for int32 biases), and have the script also emit
`snn_params_pkg.sv` with the thresholds taken from `Int16SNN().threshold1/2`
(37,718,198 and 164,691). Thresholds are **not** in the metadata JSON, so they
must not be re-derived independently.

Pixels: the quantized pixel code `rint(x*255)` equals the raw MNIST byte
exactly (0 differences on 10,000 test images), so hardware takes the raw
0..255 byte directly.

# Arithmetic Width Analysis

Signed bit widths. "Generic-safe" holds for any int16 weights and int32 bias
in this format; "proven" uses the actual exported weights.

| Quantity | Generic-safe | Proven (current weights) | Observed (10k test images) |
|---|---|---|---|
| Product pixel x weight | 24 (<= 8,355,840 < 2^23) | - | - |
| L1 sum of 784 products | 34 | - | - |
| ACC1 (+ int32 bias) | 35 | 31 | 30 |
| MEM1 | 38 | 35 | 34 |
| ACC2 | 33 | 21 | 19 |
| MEM2 | 36 | 26 | 23 |

Re-checked with the `pre_spike` path restored (`fpga/scripts/lif_width_check.py`,
interval propagation over 25 steps; `decayed` is within +-(0.95|M|+0.5),
`base = decayed + cur`, `pre_spike` and `mem_new` both lie in
`[base_min - thr, base_max]`):

| Signal | L1 generic | L1 proven | L2 generic | L2 proven |
|---|---|---|---|---|
| `cur` | 35 | 31 | 33 | 21 |
| `decayed` | 38 | 35 | 36 | 25 |
| `base` | 38 | 35 | 36 | 26 |
| `pre_spike` | 38 | 35 | 36 | 26 |
| `mem_new` | 38 | 35 | 36 | 26 |
| threshold (positive) | 27 | 27 | 19 | 19 |

Membrane interval bound (generic L1): [-1.26e11, 1.26e11] (< 2^37). The
threshold is far smaller than the membrane range, and the subtraction
`base - thr` is already included in the `pre_spike`/`mem_new` bounds, so it
cannot overflow when those signals use the widths above. The input to the
decay block, `x = 19|m|`, stays below 2^42 (needs < 2^46 for the exact
reciprocal variant). Widths are unchanged from the earlier table; restoring
`previous_reset` does not widen anything. Partial sums never exceed the final
bound, so the bias may be added at any point without intermediate overflow.

Recommendation: use the generic-safe widths as parameters, plus a generator
assertion that the proven bounds hold for the exported files. Tight widths are
a later optimisation.

Counters: spike count 5 b (0..25); `time_step` 5 b; L1 pixel index 10 b; L1
neuron 8 b; L1 weight address 18 b; L2 input 8 b; L2 neuron 4 b; L2 weight
address 12 b; argmax index 4 b; clear counter 8 b.

# LIF Hardware Mapping

Reference (`Int16SNN._lif_step`, thresholds in accumulator units):

```
pr      = mem > thr
decayed = round_ties_to_even(19*mem/20)
pre     = decayed + cur - pr*thr
spike   = pre > thr                 (strict)
mem_new = pre - (spike - pr)*thr
```

**Correction (supersedes the earlier draft).** Only the *final membrane*
simplifies: `mem_new = (decayed + cur) - spike*thr`, because the `pr*thr`
terms cancel. **`previous_reset` is NOT removable from the spike decision.**
`spike = (decayed + cur - pr*thr) > thr` differs from
`spike = (decayed + cur) > thr` whenever `pr = 1`.
Example: thr=100, mem=120, decayed=114, cur=50, pr=1: base=164,
pre_spike=64, spike=0; the rejected form would give spike=1. Measured with
the real thresholds, the rejected form picks the wrong spike in about 4 % of
random vectors (16,742 of 400,282 for layer 1), so it is invalid.

**Hardware equations (all comparisons strict `>`, exactly as Python):**

```
prev_reset = mem > thr                    # comparator on the stored membrane
decayed    = exact round_ties_to_even(19*mem/20)       # decay block
base       = decayed + cur
pre_spike  = base - (prev_reset ? thr : 0)  # conditional subtract #1
spike      = pre_spike > thr                # strict comparator
mem_new    = base - (spike ? thr : 0)       # conditional subtract #2
```

Datapath of one `lif_core` (comparator 1 and the decay block run in parallel):

```
mem --+--> [ > thr ] ------------------------> prev_reset
      |                                            |
      +--> [ decay 19/20, ties-even ] --> decayed  |
cur ---------------------------------> [ + ] --> base
                       base --> [ - (prev_reset?thr:0) ] --> pre_spike
                       pre_spike --> [ > thr ] --> spike  (-> counter / s1[j])
                       base --> [ - (spike?thr:0) ] --> mem_new --> mem RAM
```

Verified by `fpga/scripts/lif_equiv_check.py` (see Verification Results
below). Note that `pre_spike` is used only for the comparison; it is not
stored (the stored value is `mem_new`).

**Decay by 19/20 with ties-to-even** (sign-magnitude, no divider):

```
a = |m|;  x = 19a = (a<<4) + (a<<1) + a          (< 2^42 for 38-bit m)
q = (x * M) >> 48,   M = ceil(2^48/20) = 14,073,748,835,533
r = x - 20q                                      (0..19)
inc = (r > 10) | (r == 10 & q[0])                (ties to even)
mag = q + inc;   decayed = m<0 ? -mag : mag
```

Exactness: `M*20 - 2^48 = 4`, and the reciprocal error stays below the
`1/20` fractional margin for `x < 2^46`. A Python model of exactly this
algorithm was checked bit-exact against `Int16SNN`: 0 mismatches in 600,011
decay cases (random to +-2^37, all ties `20k+10`, edges) and 0 mismatches in
300,000 full LIF steps for each of thr = 37,718,198 / 164,691 / 12,345 / 1.
### Decay implementation options and realistic DSP cost

The reciprocal multiply above is **not** a one-DSP operation. A 42-bit by
44-bit unsigned product on a DSP48E1 (25x18 signed = 24x17 unsigned) needs
ceil(42/24) x ceil(44/17) = 2 x 3 = **6 DSP slices** (DSP48E2 27x18: also 6),
plus an adder tree and a pipeline of ~4-5 stages. The earlier "about 1 DSP"
figure counted only the layer-1 MAC and was incomplete.

| Option | DSP | LUT / logic | Latency | Notes |
|---|---|---|---|---|
| A. Reciprocal constant multiplier (42x44) | 6 per instance (6 total if one decay unit is shared by both layers) | adder tree | 4-5 stages | Exact for x < 2^46. M has 23 set bits, so a LUT-only constant multiplier is possible but large |
| B. Shift/add + exact divide-by-5 (x=19a, y=x>>2, q~y/5 by shifts, fix-up) | **0** | ~12 wide adders + 4 comparators | 3-4 pipeline stages | Bit-exact, verified |
| C. Sequential restoring divider | 0 | small | ~42 cycles/neuron update (about +300 k cycles/image, roughly 2x slower) | Smallest, slowest |
| D. **Recommended:** option B | 0 | see B | 3-4 stages, 1 result/clock | No DSP competition with the L1 MAC |

Option B/D in detail (all integer, no multiplier):

```
a = |m|;  x = 19a = (a<<4)+(a<<1)+a;  y = x >> 2        # floor(x/20) = floor(y/5)
q = (y>>1)+(y>>2); q += q>>4; q += q>>8; q += q>>16; q += q>>32; q >>= 2
q += (y-5q >= 5) + (y-5q >= 10) + (y-5q >= 15) + (y-5q >= 20)  # fix-up
r = ((y - 5q) << 2) | x[1:0]                           # = x mod 20, 0..19
inc = (r > 10) | (r == 10 & q[0]);  mag = q + inc;  decayed = m<0 ? -mag : mag
```

The approximation never exceeds the true quotient; the largest fix-up
observed was 10 (< 15), so the comparators cover the range. Correctness is
not traded for DSPs: the tie rule is applied on the exact remainder.
Overall DSP budget: **1 (L1 MAC) with option D**, or 7 (1 + 6) with option A
(if a shared reciprocal unit is used; 13 if each `lif_core` has its own).

## Verification Results (Python equivalence of the hardware equations)

`py -m fpga.scripts.lif_equiv_check` compares `hw_lif` against
`Int16SNN._lif_step` bit-for-bit (membrane and spike). **Correction:** an
earlier revision of the script accepted a `decay` argument but `hw_lif`
always called the reciprocal `hw_decay`, so the "shift/add full-LIF" result
did not actually exercise the shift/add decay. `hw_lif` now takes a required
`decay_function`; the harness wraps it with a call counter and asserts it was
invoked once per vector, and a deliberately wrong (sabotaged) decay is
confirmed to be detected. Both decay implementations are now run through the
full LIF on the same vectors:

| Test set (same vectors for both decays) | Vectors | Mismatches, reciprocal decay | Mismatches, shift/add decay |
|---|---|---|---|
| Layer-1 threshold 37,718,198 | 400,282 | **0** | **0** |
| Layer-2 threshold 164,691 | 400,282 | **0** | **0** |
| thr = 12,345 | 400,282 | **0** | **0** |
| thr = 1 | 400,282 | **0** | **0** |
| Decay alone, shift/add vs reference rounding (ties, +-2^k edges, random to 2^38) | 410,222 | - | **0** |

Vectors cover positive and negative membranes, mem == thr, thr-1, thr+1, 2thr,
`pre_spike` exactly at thr-1 / thr / thr+1 with `pr = 1`, positive and negative
currents, spike and no-spike (about 148 k spikes, 252 k non-spikes per set),
all `20k+10` ties, and random values. The same run shows the rejected
simplification mispredicting spikes (16,742 / 16,764 / 16,659 / 8,407 cases).

Hidden spikes are produced and consumed within the same time step; the
`previous_reset` term is derived from the stored membrane (`mem > thr`). Output LIF uses
the same `lif_core` with the layer-2 threshold. Argmax ties resolve to the
lowest index, matching `torch.argmax` (tested).

# Controller / FSM

| State | Action | Counters |
|---|---|---|
| IDLE | wait `start` | - |
| CLEAR | zero `mem1`, `mem2`, spike counters (256 cycles) | clear ctr |
| L1C_MAC | read pixel and W1, multiply, accumulate (784 cycles) | pixel i, addr1++ |
| L1C_DRAIN | flush 2 pipeline stages | - |
| L1C_STORE | add bias b1[j], write `cur1[j]`; next j or exit | neuron j |
| TS_START | begin time step | - |
| L1_LIF | RD / PIPE / WB: read `mem1[j]`,`cur1[j]`, `lif_core`, write `mem1[j]`, set `s1[j]` (8 cycles) | j |
| L2_ACC | for each k, 256 gated adds of `W2[k][j]` by `s1[j]` | k, j, addr2++ |
| L2_DRAIN | flush | - |
| L2_LIF | add b2[k], `lif_core`, write `mem2[k]`, increment counter if spike | k |
| L2_NEXT | next k or finish step | - |
| TS_DONE | `time_step++`; loop if < 25 | time_step |
| ARGMAX | scan 10 counters, strict `>` (10 cycles) | idx |
| DONE | raise `done`, hold `pred` | - |

State persisting across steps: `mem1[256]`, `mem2[10]`, the 10 counters,
`time_step`. Persisting per image: pixels and `cur1`.

# Processing Schedule

```
per image:  CLEAR -> for j in 0..255: L1 current -> cur1[j]
            for t in 0..24:
                for j: L1 LIF   (-> s1[j])
                for k in 0..9: L2 accumulate over s1, L2 LIF, count
            ARGMAX -> DONE
```

Cycles per image (cached, recommended):

```
T = 256                              (clear)
  + 256*(784+3)                      (L1 current)   = 201,472
  + 25*[256*8 + 10*(256+1+8) + 2]    (25 steps)     = 117,500
  + 10 + 3                           (argmax, start/done)
  = 319,241 cycles
```

At 100 MHz about 3.2 ms/image. Baseline recomputing L1 every step:
`25*[256*(787+8) + 2,650 + 2] + 269 = 5,154,569` cycles (~16x slower). The
8-cycle LIF budget is conservative; L2 could be ~6 (saves ~500 cycles).

# Verification Strategy

The oracle is `Int16SNN`, not the float model. A trace generator in
`fpga/scripts` re-runs the loop calling `Int16SNN._lif_step` (without modifying
it), checks final counts against `Int16SNN`, and per image exports pixels,
`cur1`, per-step `mem1`/`spikes1`, `cur2`/`mem2`/`spikes2`, counts and the
prediction.

Levels:

1. Module: `tb_decay` and `tb_lif_core` (ties, negatives, threshold edges),
   `tb_mac`, L2 gated accumulator.
2. Layer: `tb_l1_current` against exported `cur1`.
3. Full network: `tb_snn_top` over the first ~100 test images plus all-zero,
   all-255, and tie cases, comparing every intermediate trace and the
   prediction.
4. On-board: same vectors via the chosen input interface.

Planned test benches: `tb_decay`, `tb_lif_core`, `tb_mac`, `tb_l1_current`,
`tb_snn_top`.

# RTL Milestone 1 — Exact 19/20 Decay

Files: `fpga/rtl/decay_19_20.sv`, `fpga/tb/tb_decay_19_20.sv`,
`fpga/scripts/generate_decay_vectors.py`,
`fpga/tb/vectors/decay_19_20_vectors.txt`.

**1. Purpose.** `decay_19_20` computes the leak step of the LIF neuron,
`decayed = round_ties_to_even(19 * membrane / 20)`, bit-exactly as
`Int16SNN._lif_step` does. It is the first RTL block; the LIF core, MAC, ROMs,
controller and top level are not written yet. **This is the correctness /
reference version: purely combinational, no pipeline registers.** Pipelining
for timing comes later and must not change the arithmetic.

**2. Equation.** `decayed = RNE(19 m / 20)`, where RNE rounds to the nearest
integer and resolves exact halves to the even integer.

**3. Why 19/20.** The trained network uses beta = 0.95 = 19/20, and the integer
reference applies it exactly in integer arithmetic. The hardware must use the
same rational number, not a nearby cheap constant.

**4. Why truncation is not sufficient.** `m - (m >> 4)` is beta = 0.9375, not
0.95, and a floor/truncating divide differs from the reference by up to one
count (and differently for negative numbers). A one-count error in a membrane
is carried into every later step and can flip a spike. The rule must be exact.

**5. Round-to-nearest, ties-to-even.** The exact value `19m/20` is rounded to
the nearest integer; if the fractional part is exactly 0.5 the result is the
neighbouring *even* integer.

**6. Positive example.** m = 11: 19*11/20 = 10.45 -> **10**.
m = 20: 19 -> **19**.

**7. Negative example.** m = -20: **-19**. m = -11: -10.45 -> **-10**.
Rounding is applied to the magnitude and the sign is restored afterwards
(sign-magnitude, matching the reference).

**8. Tie examples.** `19m/20 = N + 0.5` happens exactly when `19|m| mod 20 = 10`,
which for integers is `|m| = 20t + 10`. The achievable halves are therefore
9.5, 28.5, 47.5, 66.5, ... (the tie values 2.5 / 3.5 cannot occur as
`19m/20` for any integer m, because `19m = 50` or `70` has no integer
solution). Tested: m = 10: 9.5 -> **10**; m = 30: 28.5 -> **28**; m = 50:
47.5 -> **48**; m = 70: 66.5 -> **66**; and the negatives -10 -> -10,
-30 -> -28, -50 -> -48, -70 -> -66.

**Block diagram**

```
 membrane (signed W)
     |
     v
 absolute value (W-bit unsigned magnitude a, sign kept)
     |
     v
 x = 19*a = (a<<4) + (a<<1) + a
     |
     v
 y = x >> 2   (floor(x/20) == floor(y/5))
     |
     v
 q ~ y/5 by shifts/adds:  (y>>1)+(y>>2), then q+=q>>4, >>8, >>16, >>32, then >>2
     |
     v
 exact fix-up: r4 = y - 5q ; q += (r4>=5)+(r4>=10)+(r4>=15)+(r4>=20)
     |
     v
 remainder r = x mod 20 = {y - 5q, x[1:0]}
     |
     v
 ties-to-even: inc = (r > 10) | (r == 10 & q[0]) ; mag = q + inc
     |
     v
 restore sign (two's-complement negate if m < 0)
     |
     v
 decayed membrane (signed W)
```

**9. Width analysis** (W = MEM_WIDTH = 38; no implicit sizing is relied on,
every intermediate is declared with an explicit width):

| Signal | Type | Width (W=38) | Bound |
|---|---|---|---|
| `membrane_in` | signed | W = 38 | [-2^37, 2^37-1] |
| `a` (magnitude) | unsigned | W = 38 | <= 2^37 (needs 38 bits as unsigned) |
| `x = 19a` | unsigned | W+5 = 43 | <= 19*2^37 < 2^42 |
| `y = x>>2` | unsigned | W+3 = 41 | < 2^40 |
| `q1..q6` (approx. quotient) | unsigned | W+3 = 41 | <= y/5, never above |
| `r4 = y - 5q6` | unsigned | W+3 = 41 | observed <= 10 before fix-up |
| `fix` | unsigned | 3 | 0..4 (observed <= 2) |
| `q` | unsigned | 41 | = floor(x/20) |
| `rem_y = y - 5q` | unsigned | 41 (3 used) | 0..4 |
| `r = x mod 20` | unsigned | 5 | 0..19 |
| `mag = q + inc` | unsigned | 41 | <= 0.95*2^37 + 1 < 2^37 |
| `decayed_out` | signed | W = 38 | magnitude < 2^37 |

The shift ladder 4/8/16/32 covers y up to 2^64, so the module is valid for
MEM_WIDTH 2..60 (checked at elaboration with `$fatal`). The approximation
error is a small constant (truncation of nine shifts), so four fix-up
comparators are enough at every width.

**10. Sign handling.** The sign bit is `membrane_in[W-1]`; the magnitude is
computed as `~m + 1` in W unsigned bits; rounding works on the magnitude; the
result is negated again (`~mag + 1`) when the input was negative. Rounding on
the magnitude gives the symmetric result `RNE(-v) = -RNE(v)` that the Python
reference uses.

**11. Most-negative value.** For `-2^(W-1)` there is no positive W-bit signed
twos-complement counterpart, so a signed `abs()` would overflow back to
itself. Here the magnitude is an *unsigned* W-bit value, where `2^(W-1)` is
representable, and the test file includes `-2^37` (expected -130,567,005,798)
and `2^37-1`.

**12. Why shift/add.** The constant 19 is `16 + 2 + 1`, divide-by-5 is done
by a shift ladder plus an exact remainder fix-up, and `5q = 4q + q`. The
block therefore maps to LUT adders/comparators with **no DSP slices and no
`*` or `/` operators** (the 6-DSP reciprocal multiplier option is not used).

**13. Vector generation.** `py -m fpga.scripts.generate_decay_vectors` uses the
*real* reference `_round_ratio_ties_to_even(m*19, 20)` (not the RTL algorithm)
as the oracle and writes 49,104 lines of `input expected` (38-bit
two's-complement hex) to `fpga/tb/vectors/decay_19_20_vectors.txt`. It
covers 0, +-1, +-20, threshold-related values (both layers, +-1,2,3 x thr
+-2), measured and full-range extremes (-2^37, 2^37-1), powers of two and +-1
for every bit, every integer in -2000..2000, 6,915 tie cases
(`19|m| mod 20 = 10`, including large ones), 20,000 uniform random values and
20,000 log-uniform random magnitudes.

**14. Testbench.** `tb_decay_19_20` instantiates the DUT with MEM_WIDTH=38,
runs 19 directed cases (including the ties and the most-negative value) and
then every file vector, comparing with `!==` on signed vectors. On a mismatch
it prints the index, input and expected/actual in decimal and hex; it
counts tests and failures and ends in `PASS` or `$fatal`.

**15. Simulation result (actually run).** Simulator: ModelSim - Intel FPGA
Edition 10.5b (`C:\intelFPGA_lite\17.0\modelsim_ase`). Compile 0 errors,
0 warnings.

```
cd fpga\tb
vlib work
vlog -sv ..\rtl\decay_19_20.sv tb_decay_19_20.sv
vsim -c work.tb_decay_19_20 -do "run -all; quit -f"
  directed tests : 19
  file vectors   : 49104
  total tests    : 49123
  failures       : 0
  PASS: tb_decay_19_20
```

The testbench was also shown to be able to fail: a mutant using round-half-up
instead of ties-to-even produced many mismatches (e.g. m = 30 gave 29 instead
of 28) and `FAIL`; the mutant was then deleted.
For other simulators: Verilator `verilator --binary -sv ../rtl/decay_19_20.sv
tb_decay_19_20.sv --top-module tb_decay_19_20`; Vivado
`xvlog -sv ...; xelab tb_decay_19_20; xsim tb_decay_19_20 -runall` (run from
`fpga/tb`; these two were not run).

**Synthesis / timing note (not yet measured).** No synthesis has been run. The
combinational path is an adder chain (x, six shift-adds, 5q, subtract, 4
compares, 5q, subtract, increment, negate) of up to ~43 bits, so it will not
meet a high clock rate unpipelined; pipelining at the x, q6, fix-up and
rounding boundaries is planned.

# RTL Milestone 2 — Bit-Exact LIF Core

Files: `fpga/rtl/lif_core.sv`, `fpga/tb/tb_lif_core.sv`,
`fpga/scripts/generate_lif_vectors.py`,
`fpga/tb/vectors/lif_l1_vectors.txt`, `fpga/tb/vectors/lif_l2_vectors.txt`.

**Interface.** Purely combinational and stateless (the membrane storage will
belong to higher-level logic):

```systemverilog
module lif_core #(
    parameter int     MEM_WIDTH = 38,
    parameter int     CUR_WIDTH = 35,
    parameter longint THRESHOLD = 37718198
) (
    input  logic signed [MEM_WIDTH-1:0] membrane_in,   // stored membrane
    input  logic signed [CUR_WIDTH-1:0] current_in,    // layer current (incl. bias)
    output logic signed [MEM_WIDTH-1:0] membrane_out,  // value to store
    output logic                        spike_out);
```

| | MEM_WIDTH | CUR_WIDTH | THRESHOLD |
|---|---|---|---|
| Layer 1 | 38 | 35 | 37,718,198 (`Int16SNN.threshold1`) |
| Layer 2 | 36 | 33 | 164,691 (`Int16SNN.threshold2`) |

Elaboration checks (`$fatal`): `CUR_WIDTH <= MEM_WIDTH`, and `0 < THRESHOLD <
2^(MEM_WIDTH-1)`. The threshold is an explicit signed MEM_WIDTH constant
(`THR`), and `current_in` is explicitly sign-extended by replicating its sign
bit (a generate branch handles the equal-width case).

**Equations (all comparisons strict `>`).**

```
prev_reset   = membrane_in > THR            // STORED membrane, not decayed/base
decayed      = decay_19_20(membrane_in)     // verified milestone-1 block, instantiated
base         = decayed + sext(current_in)
pre_spike    = base - (prev_reset ? THR : 0)
spike_out    = pre_spike > THR
membrane_out = base - (spike_out ? THR : 0)
```

**Algebraic simplification.** Python: `mem_new = pre_spike - (spike -
prev_reset)*thr`. Substituting `pre_spike = base - prev_reset*thr` gives
`mem_new = base - prev_reset*thr - spike*thr + prev_reset*thr = base -
spike*thr`. Only the final update simplifies; `prev_reset` is still required
in `pre_spike` before the spike comparison.

**Width handling.** `decayed`, `base`, `pre_spike`, `membrane_out` are signed
MEM_WIDTH. The Arithmetic Width Analysis shows they fit for every reachable
membrane/current (generic-safe L1 38/35, L2 36/33). The core does not
saturate; stored membranes and currents outside the analysed range can wrap,
which is why the generated vectors exclude them (see below).

**Block diagram**

```
                 +---------------+
membrane ------->| decay_19_20   |--> decayed
    |            +---------------+
    |
    +--> [mem > threshold] --> prev_reset

decayed --+
           +--> base = decayed + current
current --+
                 |
                 +-- subtract threshold if prev_reset
                 |
                 v
             pre_spike
                 |
                 v
       [pre_spike > threshold]
                 |
                 v
               spike

base -- subtract threshold if spike --> membrane_new
```

**Vector generation.** `py -m fpga.scripts.generate_lif_vectors` uses the real
`Int16SNN._lif_step` as the oracle and writes
`membrane_in current_in expected_membrane_out expected_spike` as fixed-width
two's-complement hex. Coverage per layer: membrane = 0 / thr-1 / thr / thr+1,
current 0, `pre_spike` targets thr-2 ... thr+2 with `prev_reset` both 0 and 1,
positive and negative currents and membranes, decay ties
(`|m| = 20t+10`), the extreme values of both ranges, 20,000+ random vectors
per layer in several distributions (full range, +-3 thr, +-8 thr, narrow
band around thr). Candidates whose reference intermediates (`base`,
`pre_spike`, `membrane_new`) would leave the signed MEM_WIDTH range are
excluded (350 for L1, 316 for L2), since int64 Python would not wrap but
hardware would; the width analysis shows these are unreachable.

| | Vectors | prev_reset=1 / 0 | Spikes / no spike | `pre_spike` = thr-1 / thr / thr+1 | Ties |
|---|---|---|---|---|---|
| Layer 1 | 95,380 | 40,437 / 54,943 | 33,807 / 61,573 | 60 / 61 / 60 | 5,145 |
| Layer 2 | 95,414 | 40,562 / 54,852 | 33,884 / 61,530 | 59 / 60 / 59 | 5,154 |

**Regression test (the earlier architecture bug).** `tb_lif_core` also
instantiates `lif_core` with THRESHOLD=100 (MEM 16, CUR 12) and checks
membrane=120, current=50: `decayed=114, prev_reset=1, base=164,
pre_spike=64, spike=0, membrane_new=164`. The rejected form without
`prev_reset` would give spike=1. Directed cases also check the strict
comparison on the stored membrane (100 -> no reset, 101 -> reset) and on
`pre_spike` (99, 100 -> no spike; 101 -> spike), with and without
`prev_reset`.

**Testbench.** `tb_lif_layer` (one layer, parameterised, reads one vector
file) is instantiated twice by `tb_lif_core`; on a mismatch it prints index,
inputs (decimal and hex), expected and actual membrane/spike, the threshold
and the DUT internals `decayed`, `prev_reset`, `base`, `pre_spike`
(hierarchical references; the production interface is unchanged). The top
testbench ends in `PASS` or `$fatal`.

**ModelSim results (actually run, ModelSim - Intel FPGA Edition 10.5b).**

```
cd fpga\tb
vlib work
vlog -sv ..\rtl\decay_19_20.sv ..\rtl\lif_core.sv tb_lif_core.sv
vsim -c work.tb_lif_core -do "run -all; quit -f"
  Layer 1 (MEM=38 CUR=35 THR=37718198): tests = 95380  failures = 0
  Layer 2 (MEM=36 CUR=33 THR=164691)  : tests = 95414  failures = 0
  Regression (THR=100)                : failures = 0
  PASS: tb_lif_core
```

Compile: 0 errors, 0 warnings.

**Intentional-break checks** (temporary mutants, deleted, never committed):

| Mutation | L1 failures | L2 failures | Regression failures |
|---|---|---|---|
| `spike = pre_spike >= THR` | 61 | 60 | 2 |
| `prev_reset = membrane >= THR` | 7 | 7 | 1 |
| spike decided without `prev_reset` (`decayed+current > THR`) | 4,044 | 4,087 | 3 |

All three were detected.

# RTL Milestone 3 — Layer-1 MAC

Files: `fpga/rtl/l1_mac.sv`, `fpga/tb/tb_l1_mac.sv`,
`fpga/scripts/generate_mac_vectors.py`,
`fpga/tb/vectors/mac_prim_vectors.txt`, `fpga/tb/vectors/mac_neuron_vectors.txt`.

**Why 784 products per hidden neuron.** Every hidden neuron is connected to
all 28 x 28 = 784 input pixels, so one neuron's current is
`current[j] = bias1[j] + sum_{i=0..783} pixel[i] * weight1[j][i]`. `l1_mac`
computes this for ONE neuron, one pixel/weight pair per clock.

**Interface.**

```systemverilog
module l1_mac #(parameter int ACC_WIDTH = 35) (
    input  logic                        clk,
    input  logic                        rst,        // synchronous, active high
    input  logic                        load_bias,  // start a neuron: acc <= bias
    input  logic signed [31:0]          bias,
    input  logic                        valid,      // accumulate pixel*weight this clock
    input  logic        [7:0]           pixel,
    input  logic signed [15:0]          weight,
    output logic signed [ACC_WIDTH-1:0] acc_out,    // accumulator register
    output logic                        result_valid);
```

The MAC owns only its accumulator. **It has no 784 counter**: the future
controller owns `input_index 0..783`, so the MAC stays reusable and
independently testable (it only reacts to `load_bias` and `valid`). Priority:
`rst` > `load_bias` > `valid`; with neither, the accumulator holds.

**Numeric types (all explicit).**

| Signal | Type | Range |
|---|---|---|
| `pixel` | `logic [7:0]` | 0 ... 255 |
| `pixel_s` | `logic signed [8:0]` = `{1'b0, pixel}` | 0 ... 255 |
| `weight` | `logic signed [15:0]` | -32768 ... 32767 |
| `product` | `logic signed [23:0]` | -8,355,840 ... 8,355,585 |
| `bias` | `logic signed [31:0]` | int32 |
| `acc_out` | `logic signed [34:0]` | worst case -8,698,462,208 ... 8,698,262,287 |

Product range proof: `255 * -32768 = -8,355,840`, `255 * 32767 = 8,355,585`,
both inside the signed 24-bit range +-8,388,608. Accumulator: 784 products
reach at most +-6,550,978,560, plus a 32-bit bias gives +-8.70e9, which
needs 35 signed bits (limit +-17.18e9).

**Explicit sign extension.** Both multiplier operands are declared signed, so
SystemVerilog sign-extends the 9-bit pixel and the 16-bit weight to the
24-bit result; the pixel's leading 0 keeps it non-negative. The 24-bit
product and the 32-bit bias are sign-extended to 35 bits by replicating their
sign bits (`{{11{product[23]}}, product}`, `{{3{bias[31]}}, bias}`), never by
implicit rules. There is no saturation: arithmetic is exact within the
proven width.

**Accumulation convention (bias first).**

```
load_bias:  acc <= sext(bias)
valid:      acc <= acc + sext(pixel_s * weight)       (784 times)
```

After the clock edge that accumulates the 784th product, `acc_out =
bias + sum(products)`. Loading the bias first needs no extra adder at the end
and is mathematically identical to adding it last (no intermediate overflow
within 35 bits; the testbench checks the running value).

**Diagram**

```
 bias (int32)
     |
 sign extend
     |
     v
 accumulator register <------------------------+
     ^                                         |
     |                                         |
     +----- sign-extended 24-bit product ------+
                         ^
                         |
                signed multiplication
                  /             \
          pixel 9-bit        weight int16
          0..255             signed
```

**Clock semantics.** `acc_out` is a register updated with nonblocking
assignments, so right at the edge a testbench still reads the OLD value; the
new value exists only after the edge's NBA region. The testbench applies
stimulus 1 ns after an edge and samples 1 ns after the next edge.
`result_valid` is a registered "an accumulate happened at the previous edge",
so after the final product's edge `result_valid = 1` and `acc_out` already
holds the final current. In the 784-cycle neuron the controller simply waits
one cycle after the last pair.

**Numerical example.** bias = 100; pairs (pixel, weight) = (200, -3),
(255, 32767), (0, -32768): acc = 100, then 100 - 600 = -500, then
-500 + 8,355,585 = 8,355,085, then unchanged = 8,355,085. A pixel of 200
treated as signed 8-bit (-56) or a weight -1 treated as unsigned (65,535)
would give a wrong result (both are tested below).

**Current caching.** `generate_mac_vectors.py` verifies against the real
`Int16SNN`: it spies on `Int16SNN._lif_step` and captures the layer-1 current
fed to the LIF at every one of the 25 time steps for 20 MNIST images x 256
neurons. The current is identical for t = 0 ... 24 and equals
`pixels @ W1.T + b1`. The image and bias do not change with time, so the
layer-1 current can be computed once per image and cached; this is a
scheduling optimisation only and does not change any value.

**Vector generation.** The oracle is the exported
`layer1_weights_int16.npy` / `layer1_bias_int32.npy` with real MNIST test
pixels (raw bytes; verified identical to the reference's `rint(x*255)`) and
int64 arithmetic.

- Primitive (20,129): the required examples (0*-32768, 1*-1, 200*-1,
  255*-32768 = -8,355,840, 255*32767 = 8,355,585), a 9 x 13 grid of edge
  pixels/weights, and 20,000 random pairs.
- Complete neuron (84 tests x 784 products): image 0..7 x neurons 0, 1, 2,
  127, 128, 255 plus per image the neurons with the largest and smallest
  current, the most positive weights and the most negative weights; all-zero
  and all-255 images; two synthetic worst cases (all pixels 255 with weights
  -32768 and bias -2^31, and +32767 with bias 2^31-1); an alternating-extreme
  weight pattern. Currents span -8,698,462,208 ... 8,698,262,287, all inside
  signed 35 bits, and the running accumulator also stays inside (asserted in
  Python).

**Testbench.** `tb_l1_mac` covers: reset; bias initialisation (including
int32 extremes); one multiplication; multiple accumulations; negative and
positive weights; zero pixel; largest pixel; extreme int16 weights; `valid=0`
holding the accumulator; `load_bias` priority over `valid`; a mid-run reset;
then every primitive vector and every complete neuron. After EVERY clock it
compares the register with a 64-bit reference accumulator kept in the
testbench and flags any value outside signed 35 bits (no overflow occurred),
and after the 784th product it compares with the Python expected current.

**ModelSim results (actually run, ModelSim - Intel FPGA Edition 10.5b).**

```
cd fpga\tb
vlog -sv ..\rtl\l1_mac.sv tb_l1_mac.sv
vsim -c work.tb_l1_mac -do "run -all; quit -f"
  directed tests    : 15
  primitive tests   : 20129
  neuron tests      : 84 (x784 products)
  MAC (valid) cycles: 85992      (65,856 neuron + 20,129 primitive + 7 directed)
  total clock cycles: 106213
  overflow flags    : 0
  failures          : 0
  PASS: tb_l1_mac
```

Compile: 0 errors, 0 warnings.

**Intentional-break checks** (temporary mutants, deleted, not committed; an
unmutated control compiled from the same mutant file passed with 0 failures):

| Break | Failures detected |
|---|---|
| pixel treated as signed 8-bit (200 -> -56) | 73,118 |
| weight treated as unsigned (-1 -> 65,535) | 71,602 |
| 23-bit product (too small) | 18,801 |
| bias forgotten | 66,029 |
| result checked one clock too early (`-GCHECK_EARLY=1`) | 54,349 |

**Notes.** This is the correctness version: multiply and add share one clock
stage with no pipeline registers. The pipelined DSP version (product register
plus accumulator register, 2-cycle drain, see Processing Schedule) will keep
the same arithmetic. Not yet synthesised, so DSP mapping and timing are
unverified.

# Open Items (to resolve before RTL)

1. Target FPGA device and BRAM capacity (~100 RAMB36 needed).
2. On-board image input interface (UART / SPI / ROM) undecided.
3. Generic-safe vs proven (tight) widths.
4. Thresholds must be exported from `Int16SNN`, not recomputed.
5. **Confirm the cached-L1-current schedule** (deviates from the literal
   "recompute every step"; numerically identical).
6. Timing of the shift/add decay pipeline (3-4 stages, 42-bit adders) vs the
   clock target; the 6-DSP reciprocal multiplier remains an alternative.
7. Simulator choice (XSim, Verilator, Questa).
8. Beta is hard-coded as 19/20.
9. Add an argmax tie test vector.
10. (Resolved) The repository reorganization and the LIF correction are on `origin/main`.

