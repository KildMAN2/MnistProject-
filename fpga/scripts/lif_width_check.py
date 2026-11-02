"""Interval-based width analysis of the LIF datapath (decay, base, pre_spike, mem_new).

Run from the repository root:  python -m fpga.scripts.lif_width_check
"""

from __future__ import annotations

import math

import numpy as np

from software.quantization.quantization import Int16SNN
from software.paths import INT16_DIR

STEPS = 25


def bits(lo: float, hi: float) -> int:
    n = 1
    while not (-(1 << (n - 1)) <= lo and hi <= (1 << (n - 1)) - 1):
        n += 1
    return n


def propagate(c_lo: int, c_hi: int, thr: int):
    m_lo = m_hi = 0
    d_lo = d_hi = b_lo = b_hi = p_lo = p_hi = 0
    for _ in range(STEPS):
        d_lo, d_hi = math.floor(0.95 * m_lo - 0.5), math.ceil(0.95 * m_hi + 0.5)
        b_lo, b_hi = d_lo + c_lo, d_hi + c_hi
        # pre_spike = base - prev_reset*thr; mem_new = base - spike*thr
        p_lo, p_hi = b_lo - thr, b_hi
        m_lo, m_hi = min(m_lo, b_lo - thr), max(m_hi, b_hi)
    return (d_lo, d_hi), (b_lo, b_hi), (p_lo, p_hi), (m_lo, m_hi)


def report(name, c_lo, c_hi, thr):
    # the membrane stored is also bounded by the interval; use the final one
    (d, b, p, m) = propagate(c_lo, c_hi, thr)
    print(f"{name}: thr={thr}  cur=[{c_lo},{c_hi}] ({bits(c_lo, c_hi)}b)")
    print(f"   decayed  {bits(*d):2d}b  base {bits(*b):2d}b  pre_spike {bits(*p):2d}b  "
          f"mem_new {bits(*m):2d}b   (mem interval [{m[0]},{m[1]}])")
    print(f"   threshold magnitude needs {bits(-thr, thr)}b (fits in every datapath width above)")
    return bits(*d), bits(*b), bits(*p), bits(*m)


def main():
    model = Int16SNN(INT16_DIR)
    i32 = (-(1 << 31), (1 << 31) - 1)
    # generic: any int16 weights, any int32 bias
    c1 = (-(255 * 784 * 32768) + i32[0], 255 * 784 * 32767 + i32[1])
    c2 = (-(256 * 32768) + i32[0], 256 * 32767 + i32[1])
    print("== generic-safe ==")
    report("L1", *c1, model.threshold1)
    report("L2", *c2, model.threshold2)

    # proven for the exported weights
    w1 = model.weights1.astype(np.int64)
    w2 = model.weights2.astype(np.int64)
    b1 = model.biases1.astype(np.int64)
    b2 = model.biases2.astype(np.int64)
    p1 = 255 * np.where(w1 > 0, w1, 0).sum(1) + b1
    n1 = 255 * np.where(w1 < 0, w1, 0).sum(1) + b1
    p2 = np.where(w2 > 0, w2, 0).sum(1) + b2
    n2 = np.where(w2 < 0, w2, 0).sum(1) + b2
    print("== proven (exported weights) ==")
    report("L1", int(n1.min()), int(p1.max()), model.threshold1)
    report("L2", int(n2.min()), int(p2.max()), model.threshold2)


if __name__ == "__main__":
    main()
