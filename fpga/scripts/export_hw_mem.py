"""Export the golden int16/int32 parameters to FPGA memory files (RTL milestone 4).

Reads the canonical artifacts/int16/*.npy (nothing is requantized) and writes:
  fpga/mem/layer1_weights.mem   200704 lines, 4 hex digits  (int16 two's complement)
  fpga/mem/layer1_bias.mem         256 lines, 8 hex digits  (int32 two's complement)
  fpga/mem/layer2_weights.mem     2560 lines, 4 hex digits
  fpga/mem/layer2_bias.mem          10 lines, 8 hex digits
  fpga/rtl/snn_params_pkg.sv      constants derived from the arrays, metadata and Int16SNN
  fpga/tb/vectors/rom_reference_vectors.txt   "type addr hex" golden ROM reads
Every .mem file is read back and compared with the .npy (zero mismatches required).

Layout: PyTorch Linear [out, in], row-major (C order), never transposed:
  L1 addr = j*784 + i      L2 addr = k*256 + j

Run from the repository root:  py -m fpga.scripts.export_hw_mem
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from fpga.scripts.lif_width_check import bits, propagate
from software.paths import INT16_DIR, REPO_ROOT
from software.quantization.quantization import Int16SNN

MEM_DIR = REPO_ROOT / "fpga" / "mem"
RTL_DIR = REPO_ROOT / "fpga" / "rtl"
VEC_DIR = REPO_ROOT / "fpga" / "tb" / "vectors"

# vector "type" codes shared with tb_parameter_roms.sv
T_L1W, T_L1B, T_L2W, T_L2B = 0, 1, 2, 3


def hexw(v: int, nbits: int) -> str:
    return f"{int(v) & ((1 << nbits) - 1):0{nbits // 4}x}"


def write_mem(path, flat: np.ndarray, nbits: int) -> None:
    with open(path, "w", newline="\n") as f:
        for v in flat:
            f.write(hexw(v, nbits) + "\n")


def read_mem(path, nbits: int) -> np.ndarray:
    out = []
    digits = nbits // 4
    for line in open(path, "r"):
        s = line.strip()
        assert len(s) == digits and all(c in "0123456789abcdef" for c in s), (path, s)
        u = int(s, 16)
        out.append(u - (1 << nbits) if u >= (1 << (nbits - 1)) else u)
    return np.array(out, dtype=np.int64)


def sha256(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main() -> None:
    MEM_DIR.mkdir(parents=True, exist_ok=True)
    VEC_DIR.mkdir(parents=True, exist_ok=True)

    w1 = np.load(INT16_DIR / "layer1_weights_int16.npy")
    b1 = np.load(INT16_DIR / "layer1_bias_int32.npy")
    w2 = np.load(INT16_DIR / "layer2_weights_int16.npy")
    b2 = np.load(INT16_DIR / "layer2_bias_int32.npy")
    # scale files are loaded only to confirm they exist/are consistent; not exported
    for n in ("layer1_scale", "layer1_bias_scale", "layer2_scale", "layer2_bias_scale"):
        np.load(INT16_DIR / f"{n}.npy")
    meta = json.load(open(INT16_DIR / "quantization_metadata.json"))
    net = meta["network"]

    assert w1.dtype == np.int16 and w2.dtype == np.int16
    assert b1.dtype == np.int32 and b2.dtype == np.int32
    hidden, inputs = w1.shape
    outputs, hidden2 = w2.shape
    assert hidden == hidden2 and b1.shape == (hidden,) and b2.shape == (outputs,)
    assert (inputs, hidden, outputs, net["num_steps"]) == (
        net["input_features"], net["hidden_size"], net["output_classes"], net["num_steps"])

    flat1 = w1.flatten(order="C")
    flat2 = w2.flatten(order="C")
    d1, d2 = flat1.size, flat2.size
    assert d1 == hidden * inputs and d2 == outputs * hidden

    # layout assertions: boundaries plus random addresses
    rng = np.random.default_rng(20240607)
    js = np.concatenate([[0, 1, 127, 128, hidden - 1], rng.integers(0, hidden, 3000)])
    is_ = np.concatenate([[0, 1, inputs - 1, 0, inputs - 1], rng.integers(0, inputs, 3000)])
    for j, i in zip(js, is_):
        assert flat1[j * inputs + i] == w1[j, i]
    for k in range(outputs):
        for j in range(hidden):
            assert flat2[k * hidden + j] == w2[k, j]
    assert flat1[0] == w1[0, 0] and flat1[-1] == w1[-1, -1] and flat1[inputs] == w1[1, 0]
    # a transposed export would differ somewhere
    assert not np.array_equal(w1.T.flatten(order="C"), flat1)

    files = {
        "layer1_weights.mem": (flat1, 16, w1.flatten()),
        "layer1_bias.mem": (b1, 32, b1),
        "layer2_weights.mem": (flat2, 16, w2.flatten()),
        "layer2_bias.mem": (b2, 32, b2),
    }
    print("== .mem export and round trip ==")
    for name, (flat, nb, orig) in files.items():
        p = MEM_DIR / name
        write_mem(p, flat, nb)
        back = read_mem(p, nb)
        mism = int(np.count_nonzero(back != orig.astype(np.int64)))
        assert back.size == orig.size and mism == 0, (name, mism)
        print(f"{name:20s} lines={back.size:7d} width={nb // 4} hex  mismatches={mism}  "
              f"bytes={p.stat().st_size}")
        print(f"   sha256 {sha256(p)}")
    # explicit format examples
    assert hexw(1, 16) == "0001" and hexw(-1, 16) == "ffff" and hexw(-32768, 16) == "8000"
    assert hexw(1, 32) == "00000001" and hexw(-1, 32) == "ffffffff"

    # ---- parameter package (tied to the golden model) ----
    model = Int16SNN(INT16_DIR)
    thr1, thr2 = int(model.threshold1), int(model.threshold2)
    i32 = (-(1 << 31), (1 << 31) - 1)
    c1 = (-(255 * inputs * 32768) + i32[0], 255 * inputs * 32767 + i32[1])
    c2 = (-(hidden * 32768) + i32[0], hidden * 32767 + i32[1])
    acc1, acc2 = bits(*c1), bits(*c2)
    mem1 = bits(*propagate(*c1, thr1)[3])
    mem2 = bits(*propagate(*c2, thr2)[3])
    steps = int(net["num_steps"])
    spike_w = max(1, steps.bit_length())  # unsigned counter width for 0..steps
    addr1 = (d1 - 1).bit_length()
    addr2 = (d2 - 1).bit_length()
    assert (acc1, acc2, mem1, mem2, addr1, addr2, spike_w) == (35, 33, 38, 36, 18, 12, 5)

    pkg = f"""// AUTO-GENERATED by fpga/scripts/export_hw_mem.py - do not edit by hand.
// Constants describing the current trained model (derived from the exported
// arrays, quantization_metadata.json and Int16SNN thresholds).
package snn_params_pkg;
  localparam int INPUT_SIZE  = {inputs};
  localparam int HIDDEN_SIZE = {hidden};
  localparam int OUTPUT_SIZE = {outputs};
  localparam int NUM_STEPS   = {steps};

  // Weight ROM layout: row-major [out][in]
  //   L1 addr = neuron*INPUT_SIZE  + input_index
  //   L2 addr = output*HIDDEN_SIZE + hidden_index
  localparam int L1_WEIGHT_DEPTH      = {d1};
  localparam int L2_WEIGHT_DEPTH      = {d2};
  localparam int L1_WEIGHT_ADDR_WIDTH = {addr1};
  localparam int L2_WEIGHT_ADDR_WIDTH = {addr2};
  localparam int L1_BIAS_DEPTH        = {hidden};
  localparam int L2_BIAS_DEPTH        = {outputs};
  localparam int L1_BIAS_ADDR_WIDTH   = {(hidden - 1).bit_length()};
  localparam int L2_BIAS_ADDR_WIDTH   = {(outputs - 1).bit_length()};

  localparam int WEIGHT_WIDTH = 16;  // signed int16
  localparam int BIAS_WIDTH   = 32;  // signed int32
  localparam int PIXEL_WIDTH  = 8;   // unsigned 0..255

  // Integer thresholds from Int16SNN(INT16_DIR).threshold1/2
  localparam longint L1_THRESHOLD = {thr1};
  localparam longint L2_THRESHOLD = {thr2};

  // Generic-safe signed widths (valid for any int16 weights / int32 bias)
  localparam int L1_ACC_WIDTH = {acc1};
  localparam int L1_MEM_WIDTH = {mem1};
  localparam int L2_ACC_WIDTH = {acc2};
  localparam int L2_MEM_WIDTH = {mem2};

  localparam int SPIKE_COUNT_WIDTH = {spike_w};  // unsigned, counts 0..NUM_STEPS

  // Memory initialisation files (paths relative to the simulator working directory
  // fpga/tb; the synthesis flow must supply its own path)
  localparam string L1_WEIGHT_FILE = "../mem/layer1_weights.mem";
  localparam string L1_BIAS_FILE   = "../mem/layer1_bias.mem";
  localparam string L2_WEIGHT_FILE = "../mem/layer2_weights.mem";
  localparam string L2_BIAS_FILE   = "../mem/layer2_bias.mem";
endpackage
"""
    with open(RTL_DIR / "snn_params_pkg.sv", "w", newline="\n") as f:
        f.write(pkg)
    print(f"snn_params_pkg.sv written: thr1={thr1} thr2={thr2} widths acc {acc1}/{acc2} "
          f"mem {mem1}/{mem2} addr {addr1}/{addr2} spike {spike_w}")

    # ---- golden ROM read vectors: "type addr rawhex" ----
    vec: list[tuple[int, int, str]] = []
    l1_dir = [0, 1, 783, 784, 785, 127 * 784, 128 * 784, 255 * 784, d1 - 1, 255 * 784 + 783,
              1567, 1568, 100000, 200703]
    l1_dir += [int(a) for a in np.where(flat1 == -32768)[0][:5]]
    l1_dir += [int(a) for a in np.where(flat1 == 32767)[0][:5]]
    l1_dir += [int(a) for a in np.where(flat1 == -1)[0][:5]]
    l1_dir += [int(a) for a in np.where(flat1 == 0)[0][:5]]
    l1_dir += [int(np.argmax(flat1)), int(np.argmin(flat1))]
    l1_addrs = l1_dir + [int(a) for a in rng.integers(0, d1, 12000)]
    vec += [(T_L1W, a, hexw(flat1[a], 16)) for a in l1_addrs]
    vec += [(T_L1B, a, hexw(b1[a], 32)) for a in range(hidden)]
    vec += [(T_L2W, a, hexw(flat2[a], 16)) for a in range(d2)]
    vec += [(T_L2B, a, hexw(b2[a], 32)) for a in range(outputs)]
    with open(VEC_DIR / "rom_reference_vectors.txt", "w", newline="\n") as f:
        for t, a, h in vec:
            f.write(f"{t} {a} {h}\n")
    cnt = {t: sum(1 for v in vec if v[0] == t) for t in range(4)}
    print(f"rom_reference_vectors.txt: {len(vec)} entries  L1W={cnt[0]} L1B={cnt[1]} "
          f"L2W={cnt[2]} L2B={cnt[3]}")


if __name__ == "__main__":
    main()
