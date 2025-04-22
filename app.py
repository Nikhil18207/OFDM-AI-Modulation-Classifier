import os
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"
import streamlit as st
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import stft
from collections import Counter
from model import HybridCNNLSTM

# Label mappings
MOD_LABELS = {
    'BPSK_BPSK': 0, 'BPSK_QPSK': 1, 'BPSK_8PSK': 2,
    'QPSK_QPSK': 3, 'QPSK_BPSK': 4, 'QPSK_8PSK': 5
}
inv_label_map = {v: k for k, v in MOD_LABELS.items()}

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridCNNLSTM(num_classes=6).to(device)
model.load_state_dict(torch.load("hybrid_cnn_lstm_best.pth", map_location=device))
model.eval()

# Streamlit UI
st.set_page_config(page_title="OFDM Modulation Classifier", layout="wide")
st.title("📡 OFDM Modulation Classification - AI Demo")

uploaded_file = st.file_uploader("Upload a low-SNR .h5 signal file", type="h5")

if uploaded_file:
    st.write(f"📁 Uploaded File: `{uploaded_file.name}`")
    true_label_from_name = uploaded_file.name.split('.')[0]
    st.info(f"🧪 True Label (from filename): **{true_label_from_name}**")

    with h5py.File(uploaded_file, 'r') as f:
        key = list(f.keys())[0]
        signal = np.array(f[key]).squeeze()

    segment_length = 1024
    nperseg = 128
    predictions = []
    all_probs = []

    for i in range(len(signal) // segment_length):
        try:
            segment = signal[i * segment_length : (i+1) * segment_length]
            f, t, Zxx = stft(segment.astype(np.complex64), nperseg=nperseg)
            spectrogram = np.abs(Zxx).astype(np.float32)
            spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram) + 1e-8)
            spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(spectrogram)
                prob = torch.softmax(output, dim=1)
                pred = torch.argmax(prob, dim=1).item()
                predictions.append(pred)
                all_probs.append(prob.cpu().numpy())
        except Exception as e:
            st.warning(f"⚠️ Skipped a corrupt segment: {str(e)}")
            continue

    # Results
    counts = Counter(predictions)
    most_common = counts.most_common(1)[0]
    predicted_class = inv_label_map[most_common[0]]
    st.success(f"✅ Predicted Modulation: **{predicted_class}**")

    # Debug Table
    st.subheader("🔎 Per-Segment Prediction Breakdown (Debug)")
    segment_table = {inv_label_map[pred]: predictions.count(pred) for pred in set(predictions)}
    st.json(segment_table)

    # Bar Chart
    fig1, ax1 = plt.subplots()
    ax1.bar([inv_label_map[k] for k in counts.keys()], counts.values(), color='skyblue')
    ax1.set_title("Class Prediction Counts")
    st.pyplot(fig1)

    # Pie Chart
    fig2, ax2 = plt.subplots()
    ax2.pie(counts.values(), labels=[inv_label_map[k] for k in counts.keys()], autopct='%1.1f%%', startangle=140)
    ax2.set_title("Class Prediction Proportions")
    st.pyplot(fig2)

    # Confidence
    st.subheader("🔍 Confidence Scores (Softmax Avg across Segments)")
    avg_probs = np.mean(np.vstack(all_probs), axis=0)
    top_indices = np.argsort(avg_probs)[::-1][:3]
    for i, idx in enumerate(top_indices):
        st.write(f"**Top {i+1}: {inv_label_map[idx]}** — {avg_probs[idx]*100:.2f}%")

    fig4, ax4 = plt.subplots()
    ax4.bar([inv_label_map[i] for i in range(len(avg_probs))], avg_probs, color='orchid')
    ax4.set_ylabel("Avg Softmax Probability")
    ax4.set_title("📊 Average Confidence per Class")
    st.pyplot(fig4)

    # Spectrogram Viewer
    st.subheader("🔬 Spectrogram View of Selected Segment")
    sample_idx = st.slider("Select a segment index to view", 0, len(signal)//segment_length - 1, 0)
    segment = signal[sample_idx * segment_length : (sample_idx+1) * segment_length]
    f, t, Zxx = stft(segment.astype(np.complex64), nperseg=nperseg)
    spectrogram = np.abs(Zxx)
    fig3, ax3 = plt.subplots()
    ax3.pcolormesh(t, f, spectrogram, shading='gouraud')
    ax3.set_title("Spectrogram of Selected Segment")
    st.pyplot(fig3)

    # I/Q Plot
    st.subheader("🔍 I/Q Waveform View")
    I = segment.real
    Q = segment.imag
    fig5, (ax5, ax6) = plt.subplots(2, 1, figsize=(10, 4))
    ax5.plot(I, color='blue')
    ax5.set_title("In-phase (I) Component")
    ax5.set_ylabel("Amplitude")
    ax6.plot(Q, color='orange')
    ax6.set_title("Quadrature (Q) Component")
    ax6.set_xlabel("Time Samples")
    ax6.set_ylabel("Amplitude")
    st.pyplot(fig5)
