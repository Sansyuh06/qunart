# BDH x Tang Nano 9K — Hardware Accelerated Language Model

An end-to-end edge AI deployment running **Baby Dragon Hatchling (BDH)** causal Hebbian fast-weight attention on the **Sipeed Tang Nano 9K FPGA (Gowin GW1NR-9)**.

---

## Architecture Overview

```
[Web Chatbot UI (FastAPI + WebSocket)]
               │
               ▼
[PyTorch BDH Language Model (128-dim, 384 BPE Vocab)]
               │
  ┌────────────┴────────────┐
  │                         │
[Host CPU / CUDA]     [UART Serial Bridge]
 (Prefix Cache)             │
                            ▼
               [Tang Nano 9K FPGA Logic]
               ├── bdh_fsm.v (Command Processor)
               ├── bdh_mac_dsp.v (INT8 DSP Blocks)
               ├── bdh_zero_skipper.v (ReLU Sparsity Gating)
               └── bdh_led_chaser.v (10-bit Gamma PWM Glide)
```

---

## Features

- **Real Silicon Execution**: Streams INT8 matrix-vector projections to Gowin 18×18 DSP blocks over UART at 115,200 baud with real-time zero-skipping telemetry.
- **Fail-Loud Protection**: Prevents silent CPU fallbacks; explicitly validates bitstream responses against integer accumulator math on hardware startup.
- **Dual Offload Modes**:
  - `MODE: HEAD-ONLY FPGA (FAST UART STREAM)`: Accelerated final projection over UART (~0.15s per token).
  - `MODE: FULL FPGA (ALL 7 LINEAR LAYERS)`: Dispatches Q, K, V, Out-Proj, FC1, FC2, and Head projections to hardware DSPs.
- **Dynamic 10-bit Gamma PWM LED Rail**: Physical on-board LEDs (and synchronized web UI virtual rail) execute a continuous smooth glide crossfade animation:
  $$\text{br}_{\text{in}} = \text{fade}^2, \quad \text{br}_{\text{out}} = (1 - \text{fade})^2$$
- **Emergent Circuit Graph & Hebbian Memory Matrix**: Live visualizer rendering fast-weight associative matrix states $\rho_t = \rho_{t-1} + K_t^T V_t$.

---

## Directory Structure

```
FPGA x BDH method/
├── rtl/                        # Gowin Verilog RTL & Project Files
│   ├── bdh_accelerator.gprj   # Gowin FPGA Designer Project
│   ├── top.v                  # Top-level integration module
│   ├── bdh_fsm.v              # Hardware FSM and protocol decoder
│   ├── bdh_mac_dsp.v          # INT8 hardware multiplier/accumulator
│   ├── bdh_zero_skipper.v     # Dynamic zero-skipping gate
│   ├── bdh_led_chaser.v       # 10-bit PWM gamma crossfade engine
│   ├── uart_rx.v / uart_tx.v  # 115200 baud serial transceivers
│   └── tang_nano_9k.cst       # Physical pin constraints
├── python/                     # Python BDH Model, Driver & Tokenizer
│   ├── bdh_bpe_tokenizer.py   # Byte-level BPE tokenizer (384 vocab)
│   ├── bdh_real_llm.py        # BDH neural network & generation engine
│   ├── bdh_fpga_driver.py     # Low-level UART hardware driver
│   ├── bdh_fpga_layer.py      # PyTorch hardware-dispatched layer
│   └── bdh_llm_checkpoint.pt  # Trained 128-dim model weights
├── server/                     # Web Application Backend
│   ├── app.py                 # FastAPI server & WebSocket endpoint
│   ├── bdh_inference.py       # Asynchronous token streaming engine
│   └── fpga_bridge.py         # Hardware detection & telemetry bridge
└── static/                     # Monochromatic UI Frontend
    ├── index.html             # Responsive dashboard & chatbot
    ├── css/style.css          # Design system & LED animation styles
    └── js/                    # WebSocket, Graph & 60fps LED Chaser
```

---

## Getting Started

### 1. Flash FPGA Hardware (Tang Nano 9K)
1. Open [`rtl/bdh_accelerator.gprj`](rtl/bdh_accelerator.gprj) in **Gowin EDA**.
2. Run **Place & Route** to generate the bitstream.
3. Use **Gowin Programmer** to flash `GW1NR-9C` over USB.

### 2. Launch Local Inference Server
```powershell
pip install -r requirements.txt
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```
Open **`http://localhost:8000`** in your browser.
