"""Generate golden vectors for fpga/rtl/snn_inference_core.sv (RTL milestone 12).

Every number comes from the REAL Int16SNN: a spy on Int16SNN._lif_step captures, during the actual
forward execution, the layer-1 current/spikes and the layer-2 current/spikes of every time step for
MNIST test images 0..19 plus an all-zero and an all-255 image (22 images). The final counts equal
quantized_spike_counts(model(x)) and the prediction is np.argmax of them.

Run from the repository root:  py -m fpga.scripts.generate_full_inference_vectors

Output (fpga/tb/vectors/), image order: MNIST 0..19, all-zero, all-255:
  fi_pix.txt    22*784 lines, 2 hex digits (raw uint8 pixels, image-major)    -> $readmemh
  fi_cur1.txt   22*256 lines, 9 hex digits (35-bit two's complement cur1)      -> $readmemh
  fi_hs.txt     22*25 lines, 64 hex digits (bit j = hidden spike j, step t)    -> $readmemh
  fi_cur2.txt   22*25*10 lines, 9 hex digits (33-bit two's complement cur2)    -> $readmemh
  fi_step.txt   22*25 lines "output_spikes(3 hex) c0 .. c9" cumulative counts after step t
  fi_final.txt  22 lines "prediction c0 .. c9" (hex)
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision import datasets

from software.paths import DATA_DIR, INT16_DIR, REPO_ROOT
from software.quantization.quantization import Int16SNN, quantized_spike_counts

VEC_DIR = REPO_ROOT / "fpga" / "tb" / "vectors"
STEPS = 25
NMNIST = 20
CUR1_W, CUR2_W = 35, 33


def hexw(v: int, bits: int) -> str:
    return f"{int(v) & ((1 << bits) - 1):0{(bits + 3) // 4}x}"


def captured(model: Int16SNN, images_u8: np.ndarray):
    calls: list[tuple] = []
    original = Int16SNN._lif_step

    def spy(membrane, current, threshold):
        m_out, spk = original(membrane, current, threshold)
        calls.append((np.array(current, copy=True), np.array(spk, copy=True)))
        return m_out, spk

    Int16SNN._lif_step = staticmethod(spy)
    try:
        x = torch.tensor(images_u8, dtype=torch.float32).div(255.0).reshape(-1, 1, 28, 28)
        out = model(x)
    finally:
        Int16SNN._lif_step = staticmethod(original)
    assert len(calls) == 2 * model.num_steps
    return calls[0::2], calls[1::2], out


def main() -> None:
    model = Int16SNN(INT16_DIR)
    assert model.num_steps == STEPS
    mnist = datasets.MNIST(str(DATA_DIR), train=False, download=False).data.numpy()
    flat = mnist.reshape(10000, 784)
    images = np.concatenate([flat[:NMNIST], np.zeros((1, 784), np.uint8), np.full((1, 784), 255, np.uint8)])
    n_img = len(images)

    l1, l2, out = captured(model, images)
    counts = quantized_spike_counts(out).numpy()
    pred = np.argmax(counts, axis=1)

    pix_lines, cur1_lines, hs_lines, cur2_lines, step_lines, final_lines = [], [], [], [], [], []
    for im in range(n_img):
        cur1 = l1[0][0][im]
        for t in range(STEPS):
            # layer-1 current does not depend on the time step (cached once in hardware)
            assert np.array_equal(l1[t][0][im], cur1)
        expect_cur1 = images[im].astype(np.int64) @ model.weights1.astype(np.int64).T + model.biases1
        assert np.array_equal(expect_cur1, cur1)
        assert int(cur1.min()) >= -(1 << (CUR1_W - 1)) and int(cur1.max()) < (1 << (CUR1_W - 1))
        pix_lines += [f"{int(p):02x}" for p in images[im]]
        cur1_lines += [hexw(c, CUR1_W) for c in cur1]
        cum = np.zeros(10, np.int64)
        for t in range(STEPS):
            hs = l1[t][1][im]
            cur2 = l2[t][0][im]
            spk = l2[t][1][im].astype(np.int64)
            assert np.array_equal(spk, out[t, im].numpy())
            hs_lines.append(f"{sum(1 << j for j in range(256) if hs[j]):064x}")
            cur2_lines += [hexw(c, CUR2_W) for c in cur2]
            cum += spk
            step_lines.append(f"{sum(int(spk[k]) << k for k in range(10)):03x} " + " ".join(f"{int(c):x}" for c in cum))
        assert np.array_equal(cum, counts[im])
        final_lines.append(f"{int(pred[im]):x} " + " ".join(f"{int(c):x}" for c in counts[im]))

    assert counts[NMNIST + 1][2] == 9 and counts[NMNIST + 1][5] == 9 and pred[NMNIST + 1] == 2
    print(f"{n_img} images x {STEPS} steps from the real Int16SNN; predictions {pred.tolist()}")
    print(f"all-255 counts {counts[NMNIST + 1].tolist()} -> prediction {int(pred[NMNIST + 1])}")

    VEC_DIR.mkdir(parents=True, exist_ok=True)
    for name, lines in (("fi_pix", pix_lines), ("fi_cur1", cur1_lines), ("fi_hs", hs_lines),
                        ("fi_cur2", cur2_lines), ("fi_step", step_lines), ("fi_final", final_lines)):
        with open(VEC_DIR / f"{name}.txt", "w", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        print(f"wrote {name}.txt: {len(lines)} lines")


if __name__ == "__main__":
    main()
