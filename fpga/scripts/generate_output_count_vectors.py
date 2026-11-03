"""Generate golden vectors for fpga/rtl/output_spike_counters.sv (RTL milestone 10).

The output spikes come from the REAL Int16SNN: a spy on Int16SNN._lif_step captures the
layer-2 spike output (and the layer-1 hidden spikes, for the integration test) during the
actual forward execution of MNIST test images 0..19 plus an all-zero and an all-255 image.
The summed spikes are cross-checked against quantized_spike_counts(model(x)) and the
prediction is the argmax of those counts (first maximum on ties, like torch/numpy argmax).

Run from the repository root:  py -m fpga.scripts.generate_output_count_vectors

Output (fpga/tb/vectors/):
  oc_steps.txt  22 images x 25 lines (image-major): "spikes c0 .. c9" in hex; spikes = 3 hex
                digits (bit k = output neuron k), c = cumulative count after that step
  oc_final.txt  22 lines: "prediction c0 .. c9" (hex), final counts and Python argmax
  oc_hs.txt     INT_IMAGES x 25 lines: 64 hex digits of hidden spikes (bit j = neuron j) for
                the first INT_IMAGES images in the order of oc_steps (integration test)
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
# integration images: MNIST 0..3, the all-zero and the all-255 image
INT_IDX = [0, 1, 2, 3, NMNIST, NMNIST + 1]


def captured(model: Int16SNN, images_u8: np.ndarray):
    calls: list[tuple] = []
    original = Int16SNN._lif_step

    def spy(membrane, current, threshold):
        m_out, spk = original(membrane, current, threshold)
        calls.append((np.array(spk, copy=True), int(threshold)))
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
    assert tuple(out.shape) == (STEPS, n_img, 10)
    real_counts = quantized_spike_counts(out).numpy()          # [image, 10], real Int16SNN counts
    real_pred = np.argmax(real_counts, axis=1)

    steps_lines, final_lines, hs_lines = [], [], []
    maxcount = 0
    ties = 0
    for im in range(n_img):
        counts = np.zeros(10, np.int64)
        for t in range(STEPS):
            spk = l2[t][0][im].astype(np.int64)
            # the spy output is the same array the forward pass returned for this step
            assert np.array_equal(spk, out[t, im].numpy())
            counts += spk
            vec = sum(int(spk[k]) << k for k in range(10))
            steps_lines.append(f"{vec:03x} " + " ".join(f"{int(c):x}" for c in counts))
            if im in INT_IDX:
                hs = l1[t][0][im]
                hs_lines.append(sum(1 << j for j in range(256) if hs[j]))
        assert np.array_equal(counts, real_counts[im]), f"image {im}: counts differ from Int16SNN"
        assert counts.max() <= STEPS
        maxcount = max(maxcount, int(counts.max()))
        ties += int((counts == counts.max()).sum() > 1)
        final_lines.append(f"{int(real_pred[im]):x} " + " ".join(f"{int(c):x}" for c in counts))

    print(f"cross-check: {n_img} images x {STEPS} steps; cumulative counts equal "
          f"quantized_spike_counts(Int16SNN(x)) for every image")
    print(f"predictions: {real_pred.tolist()}")
    print(f"labels:      {datasets.MNIST(str(DATA_DIR), train=False, download=False).targets.numpy()[:NMNIST].tolist()}")
    print(f"final count rows (first 4): {real_counts[:4].tolist()}, zero image: {real_counts[NMNIST].tolist()}, "
          f"255 image: {real_counts[NMNIST + 1].tolist()}")
    print(f"max count {maxcount}, images with argmax ties: {ties}")

    VEC_DIR.mkdir(parents=True, exist_ok=True)
    with open(VEC_DIR / "oc_steps.txt", "w", newline="\n") as f:
        f.write("\n".join(steps_lines) + "\n")
    with open(VEC_DIR / "oc_final.txt", "w", newline="\n") as f:
        f.write("\n".join(final_lines) + "\n")
    with open(VEC_DIR / "oc_hs.txt", "w", newline="\n") as f:
        for v in hs_lines:
            f.write(f"{v:064x}\n")
    print(f"wrote {len(steps_lines)} step lines, {len(final_lines)} final lines, {len(hs_lines)} hidden-spike lines")


if __name__ == "__main__":
    main()
