"""Generate golden vectors for fpga/rtl/l1_mac.sv (RTL milestone 3).

Oracle: the real exported files artifacts/int16/layer1_weights_int16.npy and
layer1_bias_int32.npy, real MNIST test pixels, and the Int16SNN integer
expression  pixels @ W1.T + b1  evaluated in int64.

Run from the repository root:  py -m fpga.scripts.generate_mac_vectors

Output (fpga/tb/vectors/):
  mac_prim_vectors.txt    "pixel weight expected_product"        (hex: 2, 4, 6 digits)
  mac_neuron_vectors.txt  per test: header "bias expected_current" (hex: 8, 9 digits)
                          followed by 784 lines "pixel weight"
Also verifies that the layer-1 current does not depend on the time step.
"""

from __future__ import annotations

import numpy as np
from torchvision import datasets
import torch

from software.paths import DATA_DIR, INT16_DIR, REPO_ROOT
from software.quantization.quantization import Int16SNN

VEC_DIR = REPO_ROOT / "fpga" / "tb" / "vectors"
ACC_W = 35
ACC_LO, ACC_HI = -(1 << (ACC_W - 1)), (1 << (ACC_W - 1)) - 1
IMAGES = list(range(8))  # test-set image indices used for complete-neuron tests


def hexw(v: int, bits: int) -> str:
    return f"{v & ((1 << bits) - 1):0{(bits + 3) // 4}x}"


def primitive_vectors(rng: np.random.Generator) -> list[tuple[int, int]]:
    directed = [(0, -32768), (1, -1), (200, -1), (255, -32768), (255, 32767),
                (0, 0), (255, 0), (0, 32767), (1, 1), (1, -32768), (128, -1), (127, 1)]
    pix = [0, 1, 2, 127, 128, 199, 200, 254, 255]
    wts = [-32768, -32767, -256, -255, -2, -1, 0, 1, 2, 255, 256, 32766, 32767]
    grid = [(p, w) for p in pix for w in wts]
    rnd = list(zip(rng.integers(0, 256, 20000).tolist(), rng.integers(-32768, 32768, 20000).tolist()))
    return directed + grid + rnd


def verify_current_cache(images_u8: np.ndarray, model: Int16SNN) -> None:
    """Capture the layer-1 current actually fed to the LIF at each of the 25 steps."""
    captured: list[np.ndarray] = []
    thresholds: list[int] = []
    original = Int16SNN._lif_step

    def spy(membrane, current, threshold):
        captured.append(np.array(current, copy=True))
        thresholds.append(int(threshold))
        return original(membrane, current, threshold)

    Int16SNN._lif_step = staticmethod(spy)
    try:
        x = torch.tensor(images_u8, dtype=torch.float32).div(255.0).reshape(-1, 1, 28, 28)
        model(x)
    finally:
        Int16SNN._lif_step = staticmethod(original)
    steps = model.num_steps
    assert len(captured) == 2 * steps, (len(captured), steps)
    l1 = captured[0::2]
    assert all(t == int(model.threshold1) for t in thresholds[0::2])
    expected = images_u8.reshape(len(images_u8), -1).astype(np.int64) @ \
        model.weights1.astype(np.int64).T + model.biases1.astype(np.int64)
    for t in range(steps):
        assert np.array_equal(l1[t], l1[0]), f"layer-1 current differs at step {t}"
        assert np.array_equal(l1[t], expected), f"step {t} differs from pixels@W1+b1"
    print(f"cache check: layer-1 current identical for t=0..{steps - 1} "
          f"on {len(images_u8)} images x {expected.shape[1]} neurons, and equal to pixels@W1.T+b1")


def main() -> None:
    rng = np.random.default_rng(20260303)
    model = Int16SNN(INT16_DIR)
    w1 = np.load(INT16_DIR / "layer1_weights_int16.npy")
    b1 = np.load(INT16_DIR / "layer1_bias_int32.npy")
    assert w1.dtype == np.int16 and b1.dtype == np.int32 and w1.shape == (256, 784)
    w64, b64 = w1.astype(np.int64), b1.astype(np.int64)

    mnist = datasets.MNIST(str(DATA_DIR), train=False, download=False).data.numpy()
    assert mnist.dtype == np.uint8 and mnist.shape == (10000, 28, 28)
    flat = mnist.reshape(10000, 784)

    # pixel codes used by the reference equal the raw bytes
    ref_pix = np.rint((flat[:200].astype(np.float32) / 255.0) * 255).astype(np.int64)
    assert np.array_equal(ref_pix, flat[:200].astype(np.int64))

    # ---- step 7: current cache check ------------------------------------------
    verify_current_cache(mnist[:20], model)

    # ---- A. primitive multiplication vectors ------------------------------------
    prim = primitive_vectors(rng)
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    with (VEC_DIR / "mac_prim_vectors.txt").open("w", newline="\n") as f:
        for p, w in prim:
            prod = p * w
            assert -(1 << 23) <= prod < (1 << 23)
            f.write(f"{hexw(p, 8)} {hexw(w, 16)} {hexw(prod, 24)}\n")
    print(f"primitive vectors: {len(prim)}  (product range "
          f"{min(p * w for p, w in prim)} .. {max(p * w for p, w in prim)})")

    # ---- B. complete-neuron vectors -----------------------------------------------
    tests: list[tuple[str, np.ndarray, np.ndarray, int]] = []  # label, pixels, weights, bias

    def add(label: str, pixels: np.ndarray, weights: np.ndarray, bias: int) -> None:
        tests.append((label, pixels.astype(np.int64), weights.astype(np.int64), int(bias)))

    pos_cnt = (w1 > 0).sum(1)
    neg_cnt = (w1 < 0).sum(1)
    for img in IMAGES:
        pix = flat[img].astype(np.int64)
        cur = pix @ w64.T + b64
        picks = {0: "n0", 1: "n1", 2: "n2", 127: "n127", 128: "n128", 255: "n255",
                 int(cur.argmax()): "max-current", int(cur.argmin()): "min-current",
                 int(pos_cnt.argmax()): "most-positive-weights",
                 int(neg_cnt.argmax()): "most-negative-weights"}
        for n, why in picks.items():
            add(f"img{img}/neuron{n}/{why}", pix, w1[n], b1[n])
    add("all-zero image/neuron0", np.zeros(784), w1[0], b1[0])
    add("all-zero image/neuron255", np.zeros(784), w1[255], b1[255])
    full = np.full(784, 255)
    cur = full @ w64.T + b64
    for n in (0, int(cur.argmax()), int(cur.argmin())):
        add(f"all-255 image/neuron{n}", full, w1[n], b1[n])
    # synthetic worst cases for the 35-bit accumulator (weights/bias at int16/int32 limits)
    add("worst-case negative", full, np.full(784, -32768), -(1 << 31))
    add("worst-case positive", full, np.full(784, 32767), (1 << 31) - 1)
    # alternating extremes, mixed pixel values
    alt = np.where(np.arange(784) % 2 == 0, 32767, -32768)
    add("alternating extremes", rng.integers(0, 256, 784), alt, 12345)

    lo = hi = 0
    with (VEC_DIR / "mac_neuron_vectors.txt").open("w", newline="\n") as f:
        for label, pix, wts, bias in tests:
            cur_val = bias + int((pix * wts).sum())
            assert ACC_LO <= cur_val <= ACC_HI, (label, cur_val)
            # running accumulator never leaves the 35-bit range either
            run = bias + np.cumsum(pix * wts)
            assert ACC_LO <= run.min() and run.max() <= ACC_HI, label
            lo, hi = min(lo, cur_val), max(hi, cur_val)
            f.write(f"{hexw(bias, 32)} {hexw(cur_val, ACC_W)}\n")
            for p, w in zip(pix.tolist(), wts.tolist()):
                f.write(f"{hexw(p, 8)} {hexw(w, 16)}\n")
    print(f"neuron tests: {len(tests)} x 784 products = {len(tests) * 784} MAC cycles; "
          f"current range {lo} .. {hi} (fits signed {ACC_W} bits)")
    for label, pix, wts, bias in tests[:6]:
        print(f"   e.g. {label}: expected {bias + int((pix * wts).sum())}")
    # cross-check selected real tests against the full-matrix oracle
    for img in IMAGES[:3]:
        cur_all = flat[img].astype(np.int64) @ w64.T + b64
        for label, pix, wts, bias in tests:
            if label.startswith(f"img{img}/neuron") and np.array_equal(pix, flat[img]):
                n = int(label.split("neuron")[1].split("/")[0])
                assert cur_all[n] == bias + int((pix * wts).sum())
    print("cross-check against pixels@W1.T+b1: OK")


if __name__ == "__main__":
    main()
