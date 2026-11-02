"""Generate golden vectors for fpga/rtl/lif_core.sv.

The oracle is the REAL Int16SNN._lif_step (int64 reference), not a port of the
RTL. Run from the repository root:  py -m fpga.scripts.generate_lif_vectors

Output files (fpga/tb/vectors/), one line per vector, fixed-width two's-complement hex:
    membrane_in  current_in  expected_membrane_out  expected_spike(0/1)

Domain note: the RTL holds membrane/base/pre_spike in signed MEM_WIDTH bits. The
width analysis guarantees this for reachable values only, so vectors whose
reference intermediates (decayed+current, pre_spike, membrane_new) would leave
the signed MEM_WIDTH range are excluded (Python int64 would not wrap, hardware
would). The count of excluded candidates is printed.
"""

from __future__ import annotations

import numpy as np

from software.paths import INT16_DIR, REPO_ROOT
from software.quantization.quantization import Int16SNN, _round_ratio_ties_to_even

VEC_DIR = REPO_ROOT / "fpga" / "tb" / "vectors"


def hexw(v: int, bits: int) -> str:
    return f"{v & ((1 << bits) - 1):0{(bits + 3) // 4}x}"


def decay(m: int) -> int:
    return int(_round_ratio_ties_to_even(np.array([m * 19], dtype=np.int64), 20)[0])


def candidates(thr: int, mem_w: int, cur_w: int, rng: np.random.Generator):
    mlo, mhi = -(1 << (mem_w - 1)), (1 << (mem_w - 1)) - 1
    clo, chi = -(1 << (cur_w - 1)), (1 << (cur_w - 1)) - 1
    out: list[tuple[int, int]] = []
    mems = [0, 1, -1, thr - 1, thr, thr + 1, thr + 2, 2 * thr, 3 * thr, -thr, -thr - 1,
            thr // 2, -(thr // 2), mlo, mlo + 1, mhi, mhi - 1, 10, -10, 30, -30, 70, -70]
    mems += [20 * t + 10 for t in (0, 1, 2, 5, 1000, 12345)]
    mems += [-(20 * t + 10) for t in (0, 1, 2, 5, 1000, 12345)]
    curs = [0, 1, -1, thr - 1, thr, thr + 1, -thr, -thr - 1, clo, chi, 50, -50]
    out += [(m, c) for m in mems for c in curs]
    # pre_spike exactly thr-1 / thr / thr+1, with prev_reset = 0 and 1
    for m in mems:
        if not (mlo <= m <= mhi):
            continue
        pr = 1 if m > thr else 0
        for target in (thr - 2, thr - 1, thr, thr + 1, thr + 2):
            out.append((m, target - decay(m) + pr * thr))
            out.append((m, target - decay(m)))  # also wrong-side of the reset term
    # random vectors
    n = 30000
    out += list(zip(rng.integers(mlo, mhi + 1, n).tolist(), rng.integers(clo, chi + 1, n).tolist()))
    out += list(zip(rng.integers(-3 * thr, 3 * thr, n).tolist(), rng.integers(-2 * thr, 2 * thr, n).tolist()))
    out += list(zip(rng.integers(-8 * thr, 8 * thr, n).tolist(), rng.integers(-thr, thr, n).tolist()))
    out += list(zip(rng.integers(thr - 50, thr + 50, 5000).tolist(), rng.integers(-60, 60, 5000).tolist()))
    return out


def generate(name: str, thr: int, mem_w: int, cur_w: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    mlo, mhi = -(1 << (mem_w - 1)), (1 << (mem_w - 1)) - 1
    clo, chi = -(1 << (cur_w - 1)), (1 << (cur_w - 1)) - 1
    cands = candidates(thr, mem_w, cur_w, rng)
    cands = [(m, c) for m, c in cands if mlo <= m <= mhi and clo <= c <= chi]
    m = np.array([p[0] for p in cands], dtype=np.int64)
    c = np.array([p[1] for p in cands], dtype=np.int64)
    dec = _round_ratio_ties_to_even(m * 19, 20)
    base = dec + c
    pre = base - (m > thr).astype(np.int64) * thr
    new, spk = Int16SNN._lif_step(m, c, thr)  # the real oracle
    ok = (np.abs(base) <= mhi) & (pre >= mlo) & (pre <= mhi) & (new >= mlo) & (new <= mhi)
    ok &= (base >= mlo)
    excluded = int((~ok).sum())
    m, c, new, spk = m[ok], c[ok], new[ok], spk[ok]
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    path = VEC_DIR / f"lif_{name}_vectors.txt"
    with path.open("w", newline="\n") as f:
        for mi, ci, ni, si in zip(m.tolist(), c.tolist(), new.tolist(), spk.tolist()):
            f.write(f"{hexw(mi, mem_w)} {hexw(ci, cur_w)} {hexw(ni, mem_w)} {int(si)}\n")
    pr = m > thr
    print(f"{name}: thr={thr} MEM={mem_w} CUR={cur_w} -> {len(m)} vectors "
          f"({excluded} out-of-width candidates excluded) -> {path.name}")
    print(f"   prev_reset=1: {int(pr.sum())}, prev_reset=0: {int((~pr).sum())}, "
          f"spikes: {int(spk.sum())}, no-spike: {int((spk == 0).sum())}, "
          f"mem==thr: {int((m == thr).sum())}, mem==thr+1: {int((m == thr + 1).sum())}, "
          f"mem==thr-1: {int((m == thr - 1).sum())}")
    ps = (dec[ok] + c) - pr.astype(np.int64) * thr
    print(f"   pre_spike==thr-1: {int((ps == thr - 1).sum())}, ==thr: {int((ps == thr).sum())}, "
          f"==thr+1: {int((ps == thr + 1).sum())}, negative cur: {int((c < 0).sum())}, "
          f"negative mem: {int((m < 0).sum())}, ties: {int(((np.abs(m) * 19) % 20 == 10).sum())}")


def main() -> None:
    model = Int16SNN(INT16_DIR)
    generate("l1", int(model.threshold1), 38, 35, 20260301)
    generate("l2", int(model.threshold2), 36, 33, 20260302)


if __name__ == "__main__":
    main()
