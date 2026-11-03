"""Generate golden vectors for fpga/rtl/argmax10.sv (RTL milestone 11).

Oracle: np.argmax(counts) (first maximum on ties). The real counts come from the REAL
Int16SNN (quantized_spike_counts(model(x))) for MNIST test images 0..19 plus an all-zero and an
all-255 image; they are cross-checked against fpga/tb/vectors/oc_final.txt (milestone 10) and
against the prediction the Int16SNN classification rule gives.

Run from the repository root:  py -m fpga.scripts.generate_argmax_vectors

Output (fpga/tb/vectors/argmax_vec.txt), one line per vector, 12 hex fields:
    c0 .. c9 prediction max_count
Order: 22 real images, then NDIR directed vectors, then NRAND random vectors.
The first line of the file header is not used; the counts are in the constants below.
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision import datasets

from software.paths import DATA_DIR, INT16_DIR, REPO_ROOT
from software.quantization.quantization import Int16SNN, quantized_spike_counts

VEC_DIR = REPO_ROOT / "fpga" / "tb" / "vectors"
NMNIST = 20
NRAND = 50000
MAXC = 25


def directed() -> list[list[int]]:
    v: list[list[int]] = []
    z = [0] * 10
    v.append(list(z))
    for k in range(10):                      # unique maximum at every output
        c = [1] * 10
        c[k] = 5
        v.append(c)
        c = list(z)
        c[k] = 25
        v.append(c)
    v.append([0, 2, 5, 1, 3, 5, 0, 1, 0, 0])             # example from the spec, tie 2/5 -> 2
    v.append([9, 0, 9, 0, 0, 9, 0, 0, 0, 0])
    v.append([0, 0, 9, 0, 0, 9, 0, 0, 0, 0])             # all-255 pattern tie 2/5
    v.append([5, 0, 0, 0, 0, 0, 0, 0, 0, 5])             # tie 0 and 9 -> 0
    v.append([0, 5, 0, 0, 5, 0, 0, 0, 5, 0])             # three-way tie 1,4,8 -> 1
    v.append([7] * 10)
    v.append([25] * 10)
    v.append([0, 0, 0, 25, 0, 0, 25, 0, 25, 0])
    v.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    v.append([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    v.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    v.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    v.append([16, 0, 0, 0, 0, 0, 0, 0, 0, 0])            # 16: bit 4 only (low-4-bit mutant)
    v.append([0, 0, 0, 0, 16, 0, 0, 0, 0, 15])
    v.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 25])
    v.append([25, 24, 25, 24, 25, 24, 25, 24, 25, 24])
    v.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 9])
    v.append([3, 3, 3, 3, 3, 3, 3, 3, 3, 4])
    for a in range(10):                                  # every two-way tie (a < b), rest lower
        for b in range(a + 1, 10):
            c = [2] * 10
            c[a] = c[b] = 11
            v.append(c)
    return v


def random_vectors(rng: np.random.Generator, n: int) -> list[list[int]]:
    out: list[list[int]] = []
    for i in range(n):
        kind = i % 6
        if kind == 0:                                    # uniform
            c = rng.integers(0, MAXC + 1, 10)
        elif kind == 1:                                  # rounded normal, spike-count like
            c = np.clip(np.rint(rng.normal(6, 5, 10)), 0, MAXC).astype(int)
        elif kind == 2:                                  # sparse: few spikes
            c = rng.integers(0, 3, 10)
        elif kind == 3:                                  # deliberately duplicated maximum
            c = rng.integers(0, MAXC, 10)
            m = int(c.max())
            for k in rng.choice(10, size=int(rng.integers(2, 5)), replace=False):
                c[k] = m
        elif kind == 4:                                  # all equal
            c = np.full(10, int(rng.integers(0, MAXC + 1)))
        else:                                            # two-way tie at the top
            c = rng.integers(0, 10, 10)
            m = int(rng.integers(11, MAXC + 1))
            a, b = rng.choice(10, size=2, replace=False)
            c[a] = c[b] = m
        out.append([int(x) for x in c])
    return out


def main() -> None:
    model = Int16SNN(INT16_DIR)
    ds = datasets.MNIST(str(DATA_DIR), train=False, download=False)
    flat = ds.data.numpy().reshape(10000, 784)
    images = np.concatenate([flat[:NMNIST], np.zeros((1, 784), np.uint8), np.full((1, 784), 255, np.uint8)])
    x = torch.tensor(images, dtype=torch.float32).div(255.0).reshape(-1, 1, 28, 28)
    real_counts = quantized_spike_counts(model(x)).numpy()
    real_pred = np.argmax(real_counts, axis=1)

    # cross-check with the milestone-10 vectors (counts and predictions)
    with open(VEC_DIR / "oc_final.txt") as f:
        rows = [[int(t, 16) for t in line.split()] for line in f if line.strip()]
    assert len(rows) == len(images)
    for i, r in enumerate(rows):
        assert r[0] == int(real_pred[i]) and r[1:] == real_counts[i].tolist(), f"image {i} differs"
    labels = ds.targets.numpy()[:NMNIST]
    print(f"real images: {len(images)}; Int16SNN predictions {real_pred.tolist()}")
    print(f"predictions equal the labels for {int((real_pred[:NMNIST] == labels).sum())}/{NMNIST} MNIST images "
          f"(informational, the golden target is Int16SNN)")
    tie_idx = [i for i in range(len(images)) if (real_counts[i] == real_counts[i].max()).sum() > 1]
    print(f"real images with argmax ties: {tie_idx}; all-255 counts {real_counts[NMNIST + 1].tolist()} "
          f"-> prediction {int(real_pred[NMNIST + 1])}")
    assert real_counts[NMNIST + 1][2] == 9 and real_counts[NMNIST + 1][5] == 9 and real_pred[NMNIST + 1] == 2

    rng = np.random.default_rng(20261103)
    d = directed()
    r = random_vectors(rng, NRAND)
    allv = [list(map(int, c)) for c in real_counts] + d + r
    ntie = 0
    with open(VEC_DIR / "argmax_vec.txt", "w", newline="\n") as f:
        for c in allv:
            a = np.array(c)
            assert a.min() >= 0 and a.max() <= MAXC
            pred = int(np.argmax(a))
            ntie += int((a == a.max()).sum() > 1)
            f.write(" ".join(f"{v:x}" for v in c) + f" {pred:x} {int(a.max()):x}\n")
    print(f"wrote {len(allv)} vectors: real={len(real_counts)}, directed={len(d)}, random={len(r)}; "
          f"vectors with a tie for the maximum: {ntie}")
    print(f"NREAL={len(real_counts)} NDIR={len(d)} NRAND={len(r)}")


if __name__ == "__main__":
    main()
