"""Generate golden vectors for fpga/rtl/l2_current_engine.sv (RTL milestone 8).

Real vectors: the hidden spikes of every time step produced by the REAL
Int16SNN forward pass (layer-1 _lif_step output) for MNIST test images 0 and 1,
an all-zero image and an all-255 image, 25 time steps each. The expected
layer-2 current is
    hidden_spikes[t].astype(int64) @ weights2.astype(int64).T + biases2
and is cross-checked against the layer-2 current actually passed into
Int16SNN._lif_step during the same forward execution.

Directed vectors (all-zero, all-one, single spike 0/127/255, alternating, random
sparse and dense) use the same equation; for all-zero the result must equal
bias2 and the all-one result is checked against bias2 + row sums.

Run from the repository root:  py -m fpga.scripts.generate_l2_current_vectors

Output (fpga/tb/vectors/):
  l2c_hs.txt   one hidden-spike vector per line, 64 hex digits (bit j = spike j)
  l2c_cur.txt  10 lines per vector, 9 hex digits (33-bit two's complement)
Vector order: 100 real (image-major, then t = 0..24), then the directed vectors.
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision import datasets

from software.paths import DATA_DIR, INT16_DIR, REPO_ROOT
from software.quantization.quantization import Int16SNN

VEC_DIR = REPO_ROOT / "fpga" / "tb" / "vectors"
ACC_W = 33
STEPS = 25


def captured_layer2(model: Int16SNN, images_u8: np.ndarray):
    """(hidden_spikes, layer2_current) per step as seen inside the real forward pass."""
    calls: list[tuple[np.ndarray, np.ndarray]] = []
    original = Int16SNN._lif_step

    def spy(membrane, current, threshold):
        m_out, spk = original(membrane, current, threshold)
        calls.append((np.array(current, copy=True), np.array(spk, copy=True)))
        return m_out, spk

    Int16SNN._lif_step = staticmethod(spy)
    try:
        x = torch.tensor(images_u8, dtype=torch.float32).div(255.0).reshape(-1, 1, 28, 28)
        model(x)
    finally:
        Int16SNN._lif_step = staticmethod(original)
    assert len(calls) == 2 * model.num_steps
    hidden = [calls[2 * t][1] for t in range(model.num_steps)]   # layer-1 spikes [batch][256]
    cur2 = [calls[2 * t + 1][0] for t in range(model.num_steps)]  # layer-2 current [batch][10]
    return hidden, cur2


def hs_hex(vec: np.ndarray) -> str:
    value = 0
    for j, s in enumerate(vec):
        if int(s):
            value |= 1 << j
    return f"{value:064x}"


def main() -> None:
    model = Int16SNN(INT16_DIR)
    assert model.num_steps == STEPS
    w2 = np.load(INT16_DIR / "layer2_weights_int16.npy")
    b2 = np.load(INT16_DIR / "layer2_bias_int32.npy")
    assert w2.shape == (10, 256) and w2.dtype == np.int16 and b2.shape == (10,) and b2.dtype == np.int32
    w64, b64 = w2.astype(np.int64), b2.astype(np.int64)
    assert np.array_equal(w64, model.weights2.astype(np.int64))
    assert np.array_equal(b64, model.biases2.astype(np.int64))

    # generic worst-case bound for the accumulator width
    gen_hi = (2**31 - 1) + 256 * 32767
    gen_lo = -(2**31) - 256 * 32768
    assert -(1 << (ACC_W - 1)) <= gen_lo and gen_hi <= (1 << (ACC_W - 1)) - 1
    assert gen_hi > (1 << 31) - 1, "32 bits would not be enough"

    mnist = datasets.MNIST(str(DATA_DIR), train=False, download=False).data.numpy()
    flat = mnist.reshape(10000, 784)
    images = np.stack([flat[0], flat[1], np.zeros(784, np.uint8), np.full(784, 255, np.uint8)])

    hidden, cur2_real = captured_layer2(model, images)
    vectors: list[np.ndarray] = []
    expected: list[np.ndarray] = []
    for im in range(len(images)):
        for t in range(STEPS):
            hs = hidden[t][im].astype(np.int64)
            cur = hs @ w64.T + b64
            assert np.array_equal(cur, cur2_real[t][im]), f"image {im} step {t}: cur2 != real forward"
            vectors.append(hidden[t][im].astype(np.uint8))
            expected.append(cur)
    n_real = len(vectors)
    nonzero = sum(int(v.sum() > 0) for v in vectors)
    print(f"cross-check: {n_real} real vectors ({len(images)} images x {STEPS} steps), "
          f"cur2 equals the real Int16SNN layer-2 current for all {n_real * 10} outputs; "
          f"{nonzero} vectors have at least one spike")

    # ---- directed vectors ---------------------------------------------------------------
    rng = np.random.default_rng(20260811)
    directed: list[np.ndarray] = []
    directed.append(np.zeros(256, np.uint8))
    directed.append(np.ones(256, np.uint8))
    for pos in (0, 127, 128, 255):
        v = np.zeros(256, np.uint8); v[pos] = 1; directed.append(v)
    directed.append((np.arange(256) % 2 == 0).astype(np.uint8))   # 1010... (even indices)
    directed.append((np.arange(256) % 2 == 1).astype(np.uint8))   # 0101...
    for _ in range(40):
        directed.append((rng.random(256) < 0.05).astype(np.uint8))  # sparse
    for _ in range(40):
        directed.append((rng.random(256) < 0.7).astype(np.uint8))   # dense
    for v in directed:
        vectors.append(v)
        expected.append(v.astype(np.int64) @ w64.T + b64)

    assert np.array_equal(expected[n_real], b64), "all-zero vector must give bias2"
    assert np.array_equal(expected[n_real + 1], w64.sum(axis=1) + b64), "all-one vector"
    exp = np.stack(expected)
    assert exp.min() >= -(1 << (ACC_W - 1)) and exp.max() <= (1 << (ACC_W - 1)) - 1
    print(f"directed vectors: {len(directed)}; total vectors {len(vectors)}; "
          f"cur2 range [{exp.min()}, {exp.max()}] fits signed {ACC_W} bits")

    VEC_DIR.mkdir(parents=True, exist_ok=True)
    with open(VEC_DIR / "l2c_hs.txt", "w", newline="\n") as f:
        for v in vectors:
            f.write(hs_hex(v) + "\n")
    with open(VEC_DIR / "l2c_cur.txt", "w", newline="\n") as f:
        for row in exp:
            for k in range(10):
                f.write(f"{int(row[k]) & ((1 << ACC_W) - 1):09x}\n")
    print(f"wrote {len(vectors)} spike vectors and {len(vectors) * 10} expected currents")


if __name__ == "__main__":
    main()
