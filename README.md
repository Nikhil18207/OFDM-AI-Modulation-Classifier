# 📡 OFDM Modulation Classification with Deep Learning

📌 Overview

This project presents a comprehensive pipeline for classifying Orthogonal Frequency Division Multiplexing (OFDM) modulation schemes using various deep learning architectures. We explore and compare multiple approaches ranging from spectrogram-based CNNs to hybrid CNN-LSTM models, using raw I/Q signals as well as engineered features like magnitude and phase.

Dataset Link -  https://ieeexplore.ieee.org/document/9467343/algorithms?tabFilter=dataset#algorithms

# ⚙️ Environment Setup

Framework: PyTorch

GPU Acceleration: CUDA-enabled (e.g., NVIDIA RTX 3060)

All models are trained on .h5 files containing raw OFDM signals

# 🧷 Dataset Structure & Label Mapping

Dataset is structured by modulation class folders:

BPSK_BPSK, BPSK_QPSK, BPSK_8PSK, QPSK_QPSK, QPSK_BPSK, QPSK_8PSK

Each class is assigned an integer label from 0 to 5

# 📂 Data Preprocessing

Raw I/Q Loader

Parses complex signal from .h5 files

Splits signal into 1024-sample segments

Extracts In-phase (I) and Quadrature (Q) channels: [2, 1024]

Spectrogram Loader

Converts I/Q signal into complex form and applies Short-Time Fourier Transform (STFT)

Extracts magnitude spectrogram [Freq x Time]

Normalizes values for stability

I/Q + Mag/Phase Loader

Adds engineered features:

I, Q, Magnitude, and Phase → [1024, 4]

# 📊 I/Q Signal Visualization

Visual plots of individual I and Q components were used to validate signal integrity and variation across classes.

# 🌈 Spectrogram Analysis

Time-frequency spectrograms were generated using STFT to highlight frequency evolution in modulated signals. These were input to CNN-based models.

# 🧠 Model Architectures

1. SpectrogramCNN

Input: [1, F, T] spectrograms

3 Convolutional layers with ReLU + MaxPooling + Dropout

Flatten + Fully Connected layers

Final Accuracy: 84.38%

2. SpectrogramCNN (BN + Dropout Improved)

Added BatchNorm and Dropout after each conv layer

Final Accuracy: 84.38% (stable from Epoch 5 onward)

3. LSTMClassifier (Raw I/Q Sequences)

Input: [1024, 2] sequence (I and Q)

2-layer bidirectional LSTM

Final Accuracy: 71.56%

4. ImprovedLSTM (I/Q + Mag/Phase)

Input: [1024, 4] → I, Q, Magnitude, Phase

3-layer bidirectional LSTM + Dropout + Scheduler

Final Accuracy: 75.35%

5. Hybrid CNN + LSTM (Spectrogram → LSTM)

CNN to extract spatial features from spectrograms

Reshaped into sequences for LSTM

Bidirectional LSTM with 2 layers

Final Accuracy: 84.29% (Best performing hybrid model)

🏁 Performance Comparison

Model

Best Validation Accuracy

SpectrogramCNN

84.38%

SpectrogramCNN (BN + Dropout)

84.38%

LSTMClassifier (I/Q)

71.56%

ImprovedLSTM (I/Q + Mag/Phase)

75.35%

Hybrid CNN + LSTM

84.29%

✅ Key Insights

Spectrogram-based models outperform raw I/Q models in classification accuracy.

Adding engineered features like magnitude and phase boosts LSTM performance.

CNN-LSTM hybrid architecture provides the best of both spatial and temporal modeling.

🔮 Future Scope

Test with more modulation types (e.g., 16QAM, 64QAM)

Extend to multi-antenna (MIMO) systems

Apply domain adaptation for unseen environments

Deploy on edge devices with ONNX/TorchScript

🧾 Citation

If you use this work, please consider citing or referencing it in your own research. For collaborations or inquiries, feel free to reach out!

Author: Nikhil KumarProject: Deep Learning for OFDM Modulation ClassificationFramework: PyTorch, CUDA
