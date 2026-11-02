# MNIST Spiking Neural Network: Python Reference and FPGA Implementation

A fully connected Leaky Integrate-and-Fire SNN that classifies MNIST, built in
two stages.

## Task 1 — Python SNN reference model (complete)

- Train the SNN on MNIST (PyTorch + snnTorch, surrogate gradients)
- Evaluate floating-point accuracy
- Quantize weights to signed int16 (per-layer scale, round, clip)
- Provide an explicit integer reference inference (`Int16SNN`)
- Export the parameters for hardware

Network: 784 -> Linear(784, 256) -> LIF -> Linear(256, 10) -> LIF, 25 time
steps, beta 0.95, threshold 1.0, static-current input, spike counting + argmax.
Test accuracy: 97.13% (floating point and int16 integer reference).

## Task 2 — FPGA implementation (not started)

- Implement an equivalent SNN in SystemVerilog
- Load the trained int16 parameters
- Simulate
- Verify against the Python **integer** reference
- Synthesize for FPGA

See [docs/TASK2_FPGA_EXPLANATION.md](docs/TASK2_FPGA_EXPLANATION.md).

## Repository layout

```text
MnistProject-/
├── README.md
├── requirements.txt
├── .gitignore
├── software/
│   ├── paths.py                    # canonical data/checkpoint/int16 paths
│   ├── snn/
│   │   ├── model.py                # MNISTSNN (float model)
│   │   ├── dataset.py              # MNIST loaders
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── lif_demo.py
│   └── quantization/
│       ├── quantization.py         # int16 export + Int16SNN integer reference
│       └── quantize_and_compare.py
├── artifacts/
│   ├── checkpoints/
│   │   ├── mnist_snn_static.pt     # canonical trained model
│   │   └── mnist_snn.pt            # old rate-coded model (obsolete)
│   └── int16/                      # exported hardware parameters
├── fpga/                           # Task 2 (placeholders only)
│   ├── rtl/  tb/  mem/  scripts/  constraints/
├── docs/
│   ├── TASK1_EXPLANATION.md
│   └── TASK2_FPGA_EXPLANATION.md
└── prompts/
    └── TASK1_ALL_PROMPTS.txt
```

MNIST is downloaded to `data/` (git-ignored) on first use.

## Running Task 1

Install dependencies (Python 3.12 was used):

```powershell
pip install -r requirements.txt
```

Run from the repository root:

```powershell
python -m software.snn.lif_demo
python -m software.snn.train
python -m software.snn.evaluate
python -m software.quantization.quantize_and_compare
```

`train` writes `artifacts/checkpoints/mnist_snn_static.pt`; `evaluate` reports
floating-point test accuracy; `quantize_and_compare` re-exports
`artifacts/int16/` and prints floating-point vs int16 accuracy. Retraining
will change the checkpoint and exported parameters.

See [docs/TASK1_EXPLANATION.md](docs/TASK1_EXPLANATION.md) for the full
technical explanation.
