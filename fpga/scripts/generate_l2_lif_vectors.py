"""Generate golden vectors for fpga/rtl/l2_lif_sweep.sv (RTL milestone 9).

Every number comes from the REAL Int16SNN: a spy on Int16SNN._lif_step captures,
during the actual forward execution, the layer-1 spikes and the layer-2
(membrane_in, current, membrane_out, spikes, threshold) of every time step for MNIST
test 0, MNIST test 1, an all-zero and an all-255 image (25 steps each). The
trajectory is also replayed by calling the real _lif_step starting from mem2 = 0
and both must be equal. Directed layer-2 cases are produced by the real _lif_step.

Run from the repository root:  py -m fpga.scripts.generate_l2_lif_vectors

Output (fpga/tb/vectors/):
  l2l_hs.txt        100 lines (image-major, t = 0..24): 64 hex digits, bit j = hidden spike j
  l2l_trace.txt     1000 lines, order image, step, k: "cur2 mem_in mem_out spike"
                    (9 hex, 9 hex, 9 hex, 1 hex; 33/36/36-bit two's complement)
  l2l_directed.txt  1000 lines "mem_in cur mem_out spike" (9, 9, 9, 1 hex), 100 groups of 10
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision import datasets

from software.paths import DATA_DIR, INT16_DIR, REPO_ROOT
from software.quantization.quantization import Int16SNN

VEC_DIR = REPO_ROOT / "fpga" / "tb" / "vectors"
MEM_W, CUR_W = 36, 33
STEPS = 25
NDIR = 1000


def hexw(v: int, bits: int) -> str:
    return f"{int(v) & ((1 << bits) - 1):0{(bits + 3) // 4}x}"


def fits(a: np.ndarray, bits: int) -> bool:
    return int(a.min()) >= -(1 << (bits - 1)) and int(a.max()) <= (1 << (bits - 1)) - 1


def captured(model: Int16SNN, images_u8: np.ndarray):
    calls: list[tuple] = []
    original = Int16SNN._lif_step

    def spy(membrane, current, threshold):
        m_out, spk = original(membrane, current, threshold)
        calls.append((np.array(membrane, copy=True), np.array(current, copy=True),
                      np.array(m_out, copy=True), np.array(spk, copy=True), int(threshold)))
        return m_out, spk

    Int16SNN._lif_step = staticmethod(spy)
    try:
        x = torch.tensor(images_u8, dtype=torch.float32).div(255.0).reshape(-1, 1, 28, 28)
        model(x)
    finally:
        Int16SNN._lif_step = staticmethod(original)
    assert len(calls) == 2 * model.num_steps
    return calls[0::2], calls[1::2]   # layer 1, layer 2


def main() -> None:
    model = Int16SNN(INT16_DIR)
    thr2 = int(model.threshold2)
    assert model.num_steps == STEPS
    mnist = datasets.MNIST(str(DATA_DIR), train=False, download=False).data.numpy()
    flat = mnist.reshape(10000, 784)
    images = np.stack([flat[0], flat[1], np.zeros(784, np.uint8), np.full(784, 255, np.uint8)])

    l1, l2 = captured(model, images)
    for t in range(STEPS):
        assert l2[t][4] == thr2

    hs_lines, trace = [], []
    spikes_per_image = []
    for im in range(len(images)):
        mem = np.zeros(10, dtype=np.int64)
        total = 0
        for t in range(STEPS):
            hs = l1[t][3][im]
            m_in, cur, m_out, spk, th = (a[im] if isinstance(a, np.ndarray) else a for a in l2[t])
            # replay with the real _lif_step from mem2 = 0 and compare with the real forward values
            r_out, r_spk = Int16SNN._lif_step(mem, cur, thr2)
            assert np.array_equal(m_in, mem), f"image {im} step {t}: membrane input differs"
            assert np.array_equal(m_out, r_out) and np.array_equal(spk, r_spk)
            assert fits(cur, CUR_W) and fits(m_out, MEM_W) and fits(m_in, MEM_W)
            cur_check = hs.astype(np.int64) @ model.weights2.astype(np.int64).T + model.biases2
            assert np.array_equal(cur_check, cur)
            hs_lines.append(sum(1 << j for j in range(256) if hs[j]))
            for k in range(10):
                trace.append((int(cur[k]), int(m_in[k]), int(m_out[k]), int(spk[k])))
            mem = r_out
            total += int(spk.sum())
        spikes_per_image.append(total)
    print(f"cross-check: {len(images)} images x {STEPS} steps x 10 neurons equal the real forward "
          f"layer-2 trajectory; output spikes per image over 25 steps: {spikes_per_image}")

    # ---- directed layer-2 cases (real _lif_step) ---------------------------------------------
    rng = np.random.default_rng(20260912)
    ties = [20 * t + 10 for t in (0, 1, 2, 3, 7, 100, 12345, thr2 // 20 - 3, thr2 // 20)]
    mems = [thr2 - 1, thr2, thr2 + 1, thr2 + 2, 2 * thr2, 0, 1, -1, -5, -thr2, -thr2 - 1,
            -2 * thr2, thr2 // 2] + ties + [-t for t in ties]
    cases: list[tuple[int, int]] = []
    for m in mems:
        d = int(Int16SNN._lif_step(np.array([m], np.int64), np.array([0], np.int64), thr2)[0][0])
        pr = 1 if m > thr2 else 0
        boundary = [thr2 + pr * thr2 - d + k for k in (-1, 0, 1)]
        for c in [0, 1234, -1234, thr2, -thr2, thr2 // 3, -(thr2 // 3), 459238, -459238] + boundary:
            cases.append((m, c))
    while len(cases) < NDIR:
        kind = rng.integers(0, 4)
        if kind == 0:
            m = int(rng.integers(-4 * thr2, 4 * thr2))
        elif kind == 1:
            m = int((20 * int(rng.integers(-(thr2 // 10), thr2 // 10)) + 10) * rng.choice([-1, 1]))
        elif kind == 2:
            m = int(rng.integers(thr2 - 50, thr2 + 50))
        else:
            m = int(rng.integers(-thr2, thr2))
        c = int(rng.integers(-2 * thr2, 2 * thr2)) if rng.random() < 0.7 else int(rng.integers(-1000, 1000))
        cases.append((m, c))
    cases = cases[:NDIR]
    cases = [cases[i] for i in rng.permutation(len(cases))]
    dm = np.array([c[0] for c in cases], np.int64)
    dc = np.array([c[1] for c in cases], np.int64)
    d_out, d_spk = Int16SNN._lif_step(dm, dc, thr2)
    assert fits(dm, MEM_W) and fits(dc, CUR_W) and fits(d_out, MEM_W)
    print(f"directed: {len(cases)} cases, spikes={int(d_spk.sum())}, prev_reset=1: {int((dm > thr2).sum())}, "
          f"negative mem: {int((dm < 0).sum())}, negative cur: {int((dc < 0).sum())}, "
          f"tie cases: {int(((19 * np.abs(dm)) % 20 == 10).sum())}")

    VEC_DIR.mkdir(parents=True, exist_ok=True)
    with open(VEC_DIR / "l2l_hs.txt", "w", newline="\n") as f:
        for v in hs_lines:
            f.write(f"{v:064x}\n")
    with open(VEC_DIR / "l2l_trace.txt", "w", newline="\n") as f:
        for c, mi, mo, s in trace:
            f.write(f"{hexw(c, CUR_W)} {hexw(mi, MEM_W)} {hexw(mo, MEM_W)} {s:x}\n")
    with open(VEC_DIR / "l2l_directed.txt", "w", newline="\n") as f:
        for i in range(len(cases)):
            f.write(f"{hexw(dm[i], MEM_W)} {hexw(dc[i], CUR_W)} {hexw(d_out[i], MEM_W)} {int(d_spk[i]):x}\n")
    print(f"wrote {len(hs_lines)} spike vectors, {len(trace)} trace lines, {len(cases)} directed cases")


if __name__ == "__main__":
    main()
