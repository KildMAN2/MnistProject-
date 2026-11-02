"""Generate golden vectors for fpga/rtl/decay_19_20.sv.

The oracle is the REAL reference _round_ratio_ties_to_even(m * 19, 20), not a
re-implementation of the RTL algorithm.

Run from the repository root:  py -m fpga.scripts.generate_decay_vectors
Output: fpga/tb/vectors/decay_19_20_vectors.txt, one "input expected" pair per
line, both fixed-width (MEM_WIDTH bits) two's-complement hex.
"""

from __future__ import annotations

import numpy as np

from software.paths import INT16_DIR, REPO_ROOT
from software.quantization.quantization import Int16SNN, _round_ratio_ties_to_even

MEM_WIDTH = 38
HEX_DIGITS = (MEM_WIDTH + 3) // 4
LO = -(1 << (MEM_WIDTH - 1))
HI = (1 << (MEM_WIDTH - 1)) - 1
OUT_FILE = REPO_ROOT / "fpga" / "tb" / "vectors" / "decay_19_20_vectors.txt"


def build_inputs() -> list[int]:
    rng = np.random.default_rng(20260212)
    model = Int16SNN(INT16_DIR)
    vals: list[int] = [0, 1, -1, 20, -20, 10, -10, 19, -19, 21, -21]
    for thr in (model.threshold1, model.threshold2):
        for k in (1, 2, 3):
            for d in (-2, -1, 0, 1, 2):
                vals += [k * thr + d, -k * thr + d]
    # measured Layer-1 membrane extremes (10k test images) and full-range extremes
    vals += [-4694181866, 2241657787, LO, LO + 1, HI, HI - 1]
    # Layer-1 proven range (35 bit) and Layer-2 range
    vals += [-11436473523 // 4, 8348544757 // 4, -2446452, 445942]
    for k in range(0, MEM_WIDTH - 1):
        for d in (-1, 0, 1):
            vals += [(1 << k) + d, -((1 << k) + d)]
    # exhaustive small range, including many ties
    vals += list(range(-2000, 2001))
    # ties: 19*|m| mod 20 == 10  <=>  |m| = 20t + 10
    for t in list(range(0, 400)) + rng.integers(0, (HI // 20) - 1, 2000).tolist():
        vals += [20 * t + 10, -(20 * t + 10)]
    # random: uniform over the full range and log-uniform magnitudes
    vals += rng.integers(LO, HI + 1, 20000).tolist()
    bits = rng.integers(1, MEM_WIDTH, 20000)
    mags = [int(rng.integers(0, 1 << int(b))) for b in bits]
    signs = rng.integers(0, 2, 20000)
    vals += [-m if s else m for m, s in zip(mags, signs)]
    vals = [v for v in vals if LO <= v <= HI]
    return vals


def main() -> None:
    vals = build_inputs()
    arr = np.array(vals, dtype=np.int64)
    expected = _round_ratio_ties_to_even(arr * 19, 20)
    assert expected.shape == arr.shape
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    mask = (1 << MEM_WIDTH) - 1
    with OUT_FILE.open("w", newline="\n") as f:
        for m, e in zip(arr.tolist(), expected.tolist()):
            assert LO <= e <= HI
            f.write(f"{m & mask:0{HEX_DIGITS}x} {e & mask:0{HEX_DIGITS}x}\n")
    ties = int(((np.abs(arr) * 19) % 20 == 10).sum())
    print(f"wrote {len(vals)} vectors to {OUT_FILE}")
    print(f"  tie cases (19*|m| mod 20 == 10): {ties}")
    print(f"  negative inputs: {int((arr < 0).sum())}, min={arr.min()}, max={arr.max()}")


if __name__ == "__main__":
    main()
