# Deep Learning Based OFDM Modulation Classification

Attention-Enhanced Hybrid CNN-LSTM with Noise-Aware Curriculum Training

## Overview

This project classifies OFDM modulation schemes (BPSK, QPSK, 8PSK combinations) from raw I/Q signals using deep learning. It compares 5 architectures and introduces two key innovations:

1. **SE Channel Attention** — Squeeze-and-Excitation blocks for adaptive feature re-weighting
2. **Curriculum Learning** — Progressive SNR training from hard (low-SNR) to easy (high-SNR)

## Models Compared

| Model | Input | Description |
|-------|-------|-------------|
| SpectrogramCNN | STFT spectrogram | 3-layer CNN with BN + Dropout |
| Improved LSTM | I/Q + Mag + Phase | 3-layer bidirectional LSTM |
| Hybrid CNN-LSTM | STFT spectrogram | CNN + LSTM (ablation baseline) |
| Attention Hybrid | STFT spectrogram | CNN + SE Attention + LSTM |
| Curriculum Hybrid | STFT spectrogram | Attention Hybrid + curriculum training |

## Key Features

- **File-level train/test split** — no data leakage from segment-level splitting
- **SNR-wise evaluation** — accuracy curves from -10 dB to +20 dB
- **Confusion matrices** at low, mid, and high SNR
- **Grad-CAM visualization** — interpretability of model decisions
- **Classification reports** — precision, recall, F1 per class
- **Complexity analysis** — parameter counts, training time, inference latency

## Dataset

IEEE OFDM Modulation Classification Dataset ([source](https://ieeexplore.ieee.org/document/9467343/algorithms?tabFilter=dataset#algorithms))

```
OFDM Modulation Classification Dataset/
├── BPSK_BPSK/   (16 .h5 files: -10dB to +20dB)
├── BPSK_QPSK/
├── BPSK_8PSK/
├── QPSK_QPSK/
├── QPSK_BPSK/
└── QPSK_8PSK/
```

6 classes, 96 files, ~4M complex samples per file, segmented into 1024-sample chunks.

## Setup

```bash
pip install -r requirements.txt
```

Requires CUDA-enabled GPU (tested on RTX 3060, RTX 4060).

## Usage

**Training (notebook):**
Open `EchonNet.ipynb` and run all cells end-to-end.

**Demo (Streamlit app):**
```bash
streamlit run app.py
```

## Project Structure

```
├── EchonNet.ipynb      # Main notebook (50 cells, end-to-end pipeline)
├── model.py            # Model definitions (SEBlock, AttentionHybridCNNLSTM, HybridCNNLSTM)
├── app.py              # Streamlit web demo
├── requirements.txt    # Python dependencies
└── .gitignore
```

## Author

Nikhil Kumar
