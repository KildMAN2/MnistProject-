"""Generate golden vectors for fpga/rtl/l1_lif_sweep.sv (RTL milestone 7).

Everything is produced by the REAL Int16SNN._lif_step() with the real integer
layer-1 threshold, starting from mem = 0 and running 25 steps on cur1 =
pixels @ W1.T + b1 (static across steps). The trajectories are cross-checked
against the layer-1 membrane/current actually seen inside Int16SNN.__call__.

Run from the repository root:  py -m fpga.scripts.generate_l1_lif_vectors

Output (fpga/tb/vectors/):
  l1l_images.txt    4*784 lines, 2 hex digits: MNIST test 0, MNIST test 1, all-zero, all-255
  l1l_cur1.txt      4*256 lines, 9 hex digits (35-bit two's complement)
  l1l_trace.txt     4*25*256 lines "mem_in mem_out spike" (10 hex, 10 hex, 1 hex), ordered
                    image, step, neuron   (mem_in of step 0 is zero)
  l1l_directed.txt  4*256 lines "mem_in cur mem_out spike" (10, 9, 10, 1 hex): directed
                    integration cases (threshold boundaries, ties, signs ...)
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision import datasets

from software.paths import DATA_DIR, INT16_DIR, REPO_ROOT
from software.quantization.quantization import Int16SNN

VEC_DIR = REPO_ROOT / "fpga" / "tb" / "vectors"
MEM_W, CUR_W = 38, 35
STEPS = 25
NSET = 4  # directed sets of 256 neurons


def hexw(v: int, bits: int) -> str:
    return f"{int(v) & ((1 << bits) - 1):0{(bits + 3) // 4}x}"


def fits(a: np.ndarray, bits: int) -> bool:
    return int(a.min()) >= -(1 << (bits - 1)) and int(a.max()) <= (1 << (bits - 1)) - 1


def captured_layer1(model: Int16SNN, images_u8: np.ndarray):
    """Layer-1 (membrane_in, current, membrane_out, spikes) per step from the real forward pass."""
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
    return calls[0::2]


def main() -> None:
    model = Int16SNN(INT16_DIR)
    thr = int(model.threshold1)
    assert model.num_steps == STEPS
    w1 = model.weights1.astype(np.int64)
    b1 = model.biases1.astype(np.int64)

    mnist = datasets.MNIST(str(DATA_DIR), train=False, download=False).data.numpy()
    flat = mnist.reshape(10000, 784)
    assert np.array_equal(
        np.rint((flat[:50].astype(np.float32) / 255.0) * 255).astype(np.int64),
        flat[:50].astype(np.int64))
    images = [flat[0], flat[1], np.zeros(784, np.uint8), np.full(784, 255, np.uint8)]
    imgs = np.stack(images)
    cur = imgs.astype(np.int64) @ w1.T + b1  # [4][256]
    assert fits(cur, CUR_W)

    # ---- golden trajectories from the real _lif_step --------------------------------
    trace = []  # [image][step] -> (mem_in, mem_out, spikes)
    spike_totals = []
    for im in range(len(images)):
        mem = np.zeros(256, dtype=np.int64)
        per_step = []
        for _ in range(STEPS):
            m_out, spk = Int16SNN._lif_step(mem, cur[im], thr)
            assert fits(m_out, MEM_W)
            per_step.append((mem.copy(), m_out.copy(), spk.copy()))
            mem = m_out
        trace.append(per_step)
        spike_totals.append(int(sum(s[2].sum() for s in per_step)))

    # ---- cross-check against the actual Int16SNN forward execution ------------------------
    real = captured_layer1(model, imgs)  # per step arrays shaped [batch=4][256]
    for t in range(STEPS):
        m_in, c, m_out, spk, th = real[t]
        assert th == thr
        for im in range(len(images)):
            assert np.array_equal(c[im], cur[im])
            assert np.array_equal(m_in[im], trace[im][t][0])
            assert np.array_equal(m_out[im], trace[im][t][1])
            assert np.array_equal(spk[im], trace[im][t][2])
    print(f"cross-check: {len(images)} images x {STEPS} steps x 256 neurons equal the real Int16SNN "
          f"layer-1 trajectory; spikes per image over 25 steps: {spike_totals}")

    # ---- directed integration cases (real _lif_step) ------------------------------------------
    rng = np.random.default_rng(20260707)
    ties = [20 * t + 10 for t in (0, 1, 2, 3, 7, 100, 12345, thr // 20 - 3, thr // 20)]
    mems = [thr - 1, thr, thr + 1, thr + 2, 2 * thr, 0, 1, -1, -5, -thr, -thr - 1, -2 * thr,
            thr // 2] + ties + [-t for t in ties]
    cases: list[tuple[int, int]] = []
    for m in mems:
        d = int(Int16SNN._lif_step(np.array([m], np.int64), np.array([0], np.int64), thr)[0][0])
        pr = 1 if m > thr else 0
        # pre_spike = decayed + cur - pr*thr  -> choose cur so pre_spike is thr-1, thr, thr+1
        boundary = [thr + pr * thr - d + k for k in (-1, 0, 1)]
        for c in [0, 12345, -12345, thr, -thr, thr // 3, -(thr // 3)] + boundary:
            cases.append((m, c))
    while len(cases) < NSET * 256:
        kind = rng.integers(0, 4)
        if kind == 0:
            m = int(rng.integers(-4 * thr, 4 * thr))
        elif kind == 1:
            m = int((20 * int(rng.integers(-(thr // 10), thr // 10)) + 10) * rng.choice([-1, 1]))
        elif kind == 2:
            m = int(rng.integers(thr - 50, thr + 50))
        else:
            m = int(rng.integers(-thr, thr))
        c = int(rng.integers(-2 * thr, 2 * thr)) if rng.random() < 0.7 else int(rng.integers(-1000, 1000))
        cases.append((m, c))
    cases = cases[: NSET * 256]
    perm = rng.permutation(len(cases))
    cases = [cases[i] for i in perm]
    dm = np.array([c[0] for c in cases], np.int64)
    dc = np.array([c[1] for c in cases], np.int64)
    d_out, d_spk = Int16SNN._lif_step(dm, dc, thr)
    assert fits(dm, MEM_W) and fits(dc, CUR_W) and fits(d_out, MEM_W)
    n_pr = int((dm > thr).sum())
    print(f"directed: {len(cases)} cases, spikes={int(d_spk.sum())}, prev_reset=1: {n_pr}, "
          f"negative mem: {int((dm < 0).sum())}, negative cur: {int((dc < 0).sum())}, "
          f"tie cases: {int(((19 * np.abs(dm)) % 20 == 10).sum())}")

    # ---- write files ----------------------------------------------------------------------
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    with open(VEC_DIR / "l1l_images.txt", "w", newline="\n") as f:
        for img in images:
            for p in img:
                f.write(f"{int(p):02x}\n")
    with open(VEC_DIR / "l1l_cur1.txt", "w", newline="\n") as f:
        for im in range(len(images)):
            for v in cur[im]:
                f.write(hexw(v, CUR_W) + "\n")
    with open(VEC_DIR / "l1l_trace.txt", "w", newline="\n") as f:
        for im in range(len(images)):
            for t in range(STEPS):
                m_in, m_out, spk = trace[im][t]
                for j in range(256):
                    f.write(f"{hexw(m_in[j], MEM_W)} {hexw(m_out[j], MEM_W)} {int(spk[j]):x}\n")
    with open(VEC_DIR / "l1l_directed.txt", "w", newline="\n") as f:
        for i in range(len(cases)):
            f.write(f"{hexw(dm[i], MEM_W)} {hexw(dc[i], CUR_W)} {hexw(d_out[i], MEM_W)} {int(d_spk[i]):x}\n")
    print(f"wrote traces: {len(images)} images x {STEPS} steps x 256 = "
          f"{len(images) * STEPS * 256} neuron updates")


if __name__ == "__main__":
    main()
