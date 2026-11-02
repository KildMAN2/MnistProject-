"""Generate golden vectors for fpga/rtl/l1_current_sweep.sv (RTL milestone 6).

For each image the full cur1[256] = pixels @ W1.T + b1 (int64) from the real
exported weights/biases and real MNIST test pixels, cross-checked against the
layer-1 current actually fed to the LIF by Int16SNN.

Run from the repository root:  py -m fpga.scripts.generate_l1_sweep_vectors

Output (fpga/tb/vectors/):
  l1s_images.txt    NIMG*784 lines, 2 hex digits (raw uint8), image-major
  l1s_expected.txt  NIMG*256 lines, 9 hex digits (35-bit two's complement), image-major
Images: MNIST test 0,1,2,3, all-zero, all-255.
"""

from __future__ import annotations

import numpy as np
from torchvision import datasets

from fpga.scripts.generate_l1_engine_vectors import captured_l1_current
from software.paths import DATA_DIR, INT16_DIR, REPO_ROOT
from software.quantization.quantization import Int16SNN

VEC_DIR = REPO_ROOT / "fpga" / "tb" / "vectors"
ACC_W = 35
ACC_LO, ACC_HI = -(1 << (ACC_W - 1)), (1 << (ACC_W - 1)) - 1
REAL = [0, 1, 2, 3]


def main() -> None:
    model = Int16SNN(INT16_DIR)
    w1 = np.load(INT16_DIR / "layer1_weights_int16.npy").astype(np.int64)
    b1 = np.load(INT16_DIR / "layer1_bias_int32.npy").astype(np.int64)
    assert w1.shape == (256, 784)

    mnist = datasets.MNIST(str(DATA_DIR), train=False, download=False).data.numpy()
    flat = mnist.reshape(10000, 784)
    assert np.array_equal(
        np.rint((flat[:50].astype(np.float32) / 255.0) * 255).astype(np.int64),
        flat[:50].astype(np.int64))

    images = [flat[i] for i in REAL]
    images.append(np.zeros(784, dtype=np.uint8))
    images.append(np.full(784, 255, dtype=np.uint8))
    imgs = np.stack(images).astype(np.int64)
    cur = imgs @ w1.T + b1  # [6][256]

    ref = captured_l1_current(model, flat[REAL])
    assert np.array_equal(ref, cur[: len(REAL)]), "oracle != Int16SNN layer-1 current"
    print(f"cross-check: all 256 currents of {len(REAL)} real images equal Int16SNN layer-1 current")
    assert cur.min() >= ACC_LO and cur.max() <= ACC_HI

    VEC_DIR.mkdir(parents=True, exist_ok=True)
    with open(VEC_DIR / "l1s_images.txt", "w", newline="\n") as f:
        for img in images:
            for p in img:
                f.write(f"{int(p):02x}\n")
    with open(VEC_DIR / "l1s_expected.txt", "w", newline="\n") as f:
        for row in cur:
            for v in row:
                f.write(f"{int(v) & ((1 << ACC_W) - 1):09x}\n")
    print(f"images: {len(images)} ({len(REAL)} real + all-zero + all-255), "
          f"{cur.size} expected currents, range {int(cur.min())}..{int(cur.max())} "
          f"(fits signed {ACC_W})")


if __name__ == "__main__":
    main()
