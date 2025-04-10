# OFDM-AI-Modulation-Classifier
AI-powered hybrid system using CNN, LSTM, and GANs to classify OFDM modulation schemes under low-SNR conditions. Enhances signal detection for SDRs, cognitive radios, and satellite communication. Trained on IEEE OFDM dataset.

Dataset Link - https://ieeexplore.ieee.org/document/9467343/algorithms?tabFilter=dataset#algorithms

# 🧩 PROGRESS SO FAR

# 📁 Step 1: Dataset Setup
Loaded the OFDM Modulation Classification Dataset from IEEE DataPort.

Parsed .h5 files across multiple SNR levels and modulation pairs.

Each .h5 file contains complex I/Q signal data with shape: (4194304, 1).

# 🧪 Step 2: Signal Preprocessing & Visualization
Created a PyTorch dataset that slices signals into [2, 1024] segments (I & Q).

Visualized:

✅ Raw I/Q waveforms

✅ Spectrograms using STFT (Short-Time Fourier Transform)

# 🧠 Step 3A: CNN for Spectrogram Classification
Designed and trained a custom CNN on spectrograms.

Achieved:

Train Accuracy: ~86.13%

Validation Accuracy: ~84.77%

Saved model as spectrogram_cnn.pth.

Built a loadable version of the model (compatible with inference).

Set up code to:

Load .h5 files

Segment & preprocess signals

Feed into the model and predict the modulation class

🔜 UP NEXT
# ✅ Step 3B: LSTM on Raw I/Q Time Series
Model to capture sequential patterns in I/Q.

Compare with CNN performance.

# ✅ Step 3C: Hybrid CNN + LSTM
CNN for feature extraction → LSTM for temporal learning.

Build the most powerful hybrid classifier.

# 🎯 Final Touch:
Create a tool to upload a signal and get the predicted modulation class.

Optional: Streamlit or CLI-based interface for easy demo.

