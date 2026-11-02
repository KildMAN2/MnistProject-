"""Generate golden vectors for fpga/rtl/l1_neuron_engine.sv (RTL milestone 5).

Oracle: pixels.astype(int64) @ weights1[neuron].astype(int64) + bias1[neuron]
using the real exported weights/biases and real MNIST test images; the value is
also cross-checked against the layer-1 current actually fed to the LIF inside
Int16SNN (captured by spying on Int16SNN._lif_step).

Run from the repository root:  py -m fpga.scripts.generate_l1_engine_vectors

Output (fpga/tb/vectors/):
  l1e_images.txt  NIMG*784 lines, 2 hex digits: raw uint8 pixels, image-major
  l1e_tests.txt   one line per test: "image_index neuron expected_current_hex9"
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision import datasets

from software.paths import DATA_DIR, INT16_DIR, REPO_ROOT
from software.quantization.quantization import Int16SNN

VEC_DIR = REPO_ROOT / "fpga" / "tb" / "vectors"
ACC_W = 35
ACC_LO, ACC_HI = -(1 << (ACC_W - 1)), (1 << (ACC_W - 1)) - 1
REAL_IMAGES = list(range(8))
FULL_SWEEP_IMAGES = (0, 1)
BOUNDARY_NEURONS = (0, 1, 2, 127, 128, 254, 255)


def captured_l1_current(model: Int16SNN, images_u8: np.ndarray) -> np.ndarray:
    """Layer-1 current exactly as fed to the LIF by the real reference (step 0)."""
    captured: list[np.ndarray] = []
    original = Int16SNN._lif_step

    def spy(membrane, current, threshold):
        captured.append(np.array(current, copy=True))
        return original(membrane, current, threshold)

    Int16SNN._lif_step = staticmethod(spy)
    try:
        x = torch.tensor(images_u8, dtype=torch.float32).div(255.0).reshape(-1, 1, 28, 28)
        model(x)
    finally:
        Int16SNN._lif_step = staticmethod(original)
    return captured[0]


def main() -> None:
    model = Int16SNN(INT16_DIR)
    w1 = np.load(INT16_DIR / "layer1_weights_int16.npy")
    b1 = np.load(INT16_DIR / "layer1_bias_int32.npy")
    assert w1.shape == (256, 784) and w1.dtype == np.int16 and b1.dtype == np.int32
    w64, b64 = w1.astype(np.int64), b1.astype(np.int64)

    mnist = datasets.MNIST(str(DATA_DIR), train=False, download=False).data.numpy()
    assert mnist.dtype == np.uint8 and mnist.shape == (10000, 28, 28)
    flat = mnist.reshape(10000, 784)
    ref_pix = np.rint((flat[:200].astype(np.float32) / 255.0) * 255).astype(np.int64)
    assert np.array_equal(ref_pix, flat[:200].astype(np.int64)), "raw bytes != round(float*255)"

    images = [flat[i] for i in REAL_IMAGES]
    images.append(np.zeros(784, dtype=np.uint8))
    images.append(np.full(784, 255, dtype=np.uint8))
    zero_idx, full_idx = len(REAL_IMAGES), len(REAL_IMAGES) + 1
    imgs = np.stack(images).astype(np.int64)
    cur = imgs @ w64.T + b64          # [NIMG][256], int64 oracle

    # cross-check against the current used by the real Int16SNN (real images only)
    ref = captured_l1_current(model, flat[REAL_IMAGES])
    assert np.array_equal(ref, cur[: len(REAL_IMAGES)]), "oracle != Int16SNN layer-1 current"
    print(f"cross-check: oracle == Int16SNN layer-1 current for {len(REAL_IMAGES)} images x 256 neurons")

    assert cur.min() >= ACC_LO and cur.max() <= ACC_HI
    tests: set[tuple[int, int]] = set()
    for im in FULL_SWEEP_IMAGES:
        tests.update((im, n) for n in range(256))
    for im in REAL_IMAGES:
        tests.update((im, n) for n in BOUNDARY_NEURONS)
        tests.add((im, int(cur[im].argmax())))
        tests.add((im, int(cur[im].argmin())))
    for im in (zero_idx, full_idx):
        tests.update((im, n) for n in BOUNDARY_NEURONS)
        tests.add((im, int(cur[im].argmax())))
        tests.add((im, int(cur[im].argmin())))
    tests_l = sorted(tests)

    VEC_DIR.mkdir(parents=True, exist_ok=True)
    with open(VEC_DIR / "l1e_images.txt", "w", newline="\n") as f:
        for img in images:
            for p in img:
                f.write(f"{int(p):02x}\n")
    with open(VEC_DIR / "l1e_tests.txt", "w", newline="\n") as f:
        for im, n in tests_l:
            v = int(cur[im, n])
            f.write(f"{im} {n} {v & ((1 << ACC_W) - 1):09x}\n")

    print(f"images: {len(images)} (real {len(REAL_IMAGES)} + all-zero + all-255)")
    print(f"tests : {len(tests_l)}  (full 256-neuron sweeps on images {FULL_SWEEP_IMAGES})")
    print(f"current range over tests: {min(int(cur[i, n]) for i, n in tests_l)} .. "
          f"{max(int(cur[i, n]) for i, n in tests_l)}  (fits signed {ACC_W})")


if __name__ == "__main__":
    main()
