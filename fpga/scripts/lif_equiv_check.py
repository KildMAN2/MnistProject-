"""Bit-exact check of the hardware LIF equations against Int16SNN._lif_step.

Run from the repository root:  python -m fpga.scripts.lif_equiv_check
"""

from __future__ import annotations

import numpy as np

from software.quantization.quantization import Int16SNN
from software.paths import INT16_DIR

RECIP_SHIFT = 48
RECIP_M = -(-(1 << RECIP_SHIFT) // 20)  # ceil(2^48 / 20)


def hw_decay(m: int) -> int:
    """round_ties_to_even(19*m/20) using shift/add and a reciprocal multiply."""
    a = abs(m)
    x = (a << 4) + (a << 1) + a
    assert x < (1 << 46)
    q = (x * RECIP_M) >> RECIP_SHIFT
    r = x - 20 * q
    assert 0 <= r < 20
    inc = 1 if (r > 10 or (r == 10 and (q & 1))) else 0
    mag = q + inc
    return -mag if m < 0 else mag


def hw_decay_shiftadd(m: int) -> int:
    """Same function with no multiplier: divide-by-5 via shifts/adds plus a fix-up."""
    a = abs(m)
    x = (a << 4) + (a << 1) + a
    y = x >> 2                       # floor(x/4); floor(y/5) == floor(x/20)
    q = (y >> 1) + (y >> 2)
    q += q >> 4
    q += q >> 8
    q += q >> 16
    q += q >> 32
    q >>= 2                          # approximate y/5, never above the true quotient
    r4 = y - 5 * q                   # small non-negative remainder
    q += (r4 >= 5) + (r4 >= 10) + (r4 >= 15) + (r4 >= 20)
    r = ((y - 5 * q) << 2) | (x & 3)  # x mod 20
    assert 0 <= r < 20
    inc = 1 if (r > 10 or (r == 10 and (q & 1))) else 0
    mag = q + inc
    return -mag if m < 0 else mag


MAX_R4 = [0]


def hw_lif(m: int, cur: int, thr: int, decay=None) -> tuple[int, int]:
    """Hardware equations, strict '>' comparisons, previous_reset kept."""
    prev_reset = m > thr
    decayed = hw_decay(m)
    base = decayed + cur
    pre_spike = base - (thr if prev_reset else 0)
    spike = pre_spike > thr
    mem_new = base - (thr if spike else 0)
    return mem_new, int(spike)


def wrong_simplified(m: int, cur: int, thr: int) -> int:
    """The rejected form: spike decided without previous_reset."""
    return int(hw_decay(m) + cur > thr)


def vectors(thr: int, rng: np.random.Generator, n_random: int, cur_lim: int):
    mem_lim = 1 << 37
    mems = [0, 1, -1, thr - 1, thr, thr + 1, 2 * thr, -thr, -thr - 1, thr // 2,
            -(thr // 2), mem_lim - 1, -mem_lim]
    # exact tie cases: 19*m = 20k + 10 (m = 20t + 10 gives 19m mod 20 = 10)
    mems += [20 * t + 10 for t in (0, 1, 2, 3, 1000, -1, -2, -3, 12345678)]
    mems += [-(20 * t + 10) for t in (0, 1, 2, 3, 1000, 12345678)]
    currents = [0, 1, -1, thr, -thr, thr - 1, thr + 1, cur_lim, -cur_lim]
    pairs = [(m, c) for m in mems for c in currents]
    # around-threshold boundary on the pre_spike decision when prev_reset = 1
    for m in (thr + 1, thr + 2, thr + 1000, 2 * thr, 3 * thr):
        d = hw_decay(m)
        for target in (thr - 1, thr, thr + 1):
            for off in (0, thr):
                pairs.append((m, target - d + off))
    ms = rng.integers(-mem_lim, mem_lim, n_random)
    cs = rng.integers(-cur_lim, cur_lim, n_random)
    ms2 = rng.integers(-3 * thr, 3 * thr, n_random)
    cs2 = rng.integers(-2 * thr, 2 * thr, n_random)
    pairs += list(zip(ms.tolist(), cs.tolist())) + list(zip(ms2.tolist(), cs2.tolist()))
    return pairs


def main() -> None:
    model = Int16SNN(INT16_DIR)
    rng = np.random.default_rng(20260211)
    total_mismatch = 0
    for name, thr, cur_lim in (
        ("layer1", model.threshold1, 1 << 33),
        ("layer2", model.threshold2, 1 << 22),
        ("thr=12345", 12345, 1 << 20),
        ("thr=1", 1, 1 << 10),
    ):
        pairs = vectors(int(thr), rng, 200_000, cur_lim)
        m = np.array([p[0] for p in pairs], dtype=np.int64)
        c = np.array([p[1] for p in pairs], dtype=np.int64)
        ref_m, ref_s = Int16SNN._lif_step(m, c, int(thr))
        hw = [hw_lif(int(a), int(b), int(thr)) for a, b in zip(m, c)]
        hw_m = np.array([h[0] for h in hw], dtype=np.int64)
        hw_s = np.array([h[1] for h in hw], dtype=np.uint8)
        bad = int((hw_m != ref_m).sum() + (hw_s != ref_s).sum())
        wrong = sum(
            wrong_simplified(int(a), int(b), int(thr)) != int(s)
            for a, b, s in zip(m, c, ref_s)
        )
        cat = {
            "prev_reset=1": int((m > thr).sum()),
            "negative mem": int((m < 0).sum()),
            "mem==thr": int((m == thr).sum()),
            "spikes": int(ref_s.sum()),
            "no-spikes": int((ref_s == 0).sum()),
        }
        print(f"{name:10s} thr={thr:>10d} vectors={len(pairs):7d} "
              f"mismatches={bad} | rejected-form spike errors={wrong} | {cat}")
        total_mismatch += bad
    # shift/add decay (no DSP): exhaustive-style check against the reference rounding
    from software.quantization.quantization import _round_ratio_ties_to_even
    dm = rng.integers(-(1 << 38), 1 << 38, 400_000).tolist()
    dm += [20 * t + 10 for t in range(-2000, 2000)] + list(range(-3000, 3000))
    dm += [(1 << k) + d for k in range(1, 38) for d in (-1, 0, 1)]
    dm += [-((1 << k) + d) for k in range(1, 38) for d in (-1, 0, 1)]
    ref = _round_ratio_ties_to_even(np.array(dm, dtype=np.int64) * 19, 20)
    bad_d = sum(hw_decay_shiftadd(v) != int(r) for v, r in zip(dm, ref))
    print(f"shift/add decay: cases={len(dm)} mismatches={bad_d}")
    total_mismatch += bad_d
    for name, thr in (("layer1", model.threshold1), ("layer2", model.threshold2), ("thr=1", 1)):
        m = rng.integers(-(1 << 37), 1 << 37, 150_000)
        c = rng.integers(-(1 << 33), 1 << 33, 150_000)
        rm, rs = Int16SNN._lif_step(m, c, int(thr))
        bad = sum(
            hw_lif(int(a), int(b), int(thr), hw_decay_shiftadd) != (int(x), int(y))
            for a, b, x, y in zip(m, c, rm, rs)
        )
        print(f"shift/add LIF {name}: mismatches={bad}")
        total_mismatch += bad
    print("TOTAL MISMATCHES:", total_mismatch)
    raise SystemExit(1 if total_mismatch else 0)


if __name__ == "__main__":
    main()
