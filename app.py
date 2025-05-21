import os
import numpy as np
import librosa
import librosa.display
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

# --- Page setup ---
st.set_page_config(page_title="Audio Deepfake Detector", layout="centered")
st.markdown(
    """
    <style>
    .main { background-color: #ffffff; }
    h1, h2, h3, label, .stButton button, .stbutton label, .stFileUploader label {
        color: #000000;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton button {
        background-color: #4b7bec;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
    .stButton button:hover {
        background-color: #3867d6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Audio and spectrogram settings ---
SR = 16000
DURATION = 2  # seconds
N_FFT = 1024
HOP_LENGTH = 512

t_alpha = 0.6
beta = 0.4

# --- Load model ---
model = tf.keras.models.load_model('retrained_final_model.h5')

# --- Signal Processing Functions ---

def compute_stft(audio):
    stft = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH))
    # Fix shape to (64, 64)
    stft_fixed = np.zeros((64, 64))
    min_rows = min(stft.shape[0], 64)
    min_cols = min(stft.shape[1], 64)
    stft_fixed[:min_rows, :min_cols] = stft[:min_rows, :min_cols]
    return stft_fixed

def compute_cqt(audio):
    cqt = np.abs(librosa.cqt(audio, sr=SR, hop_length=HOP_LENGTH))
    cqt_fixed = np.zeros((64, 64))
    min_rows = min(cqt.shape[0], 64)
    min_cols = min(cqt.shape[1], 64)
    cqt_fixed[:min_rows, :min_cols] = cqt[:min_rows, :min_cols]
    return cqt_fixed

def get_weighted_spectrogram(audio):
    stft_spec = compute_stft(audio)
    cqt_spec = compute_cqt(audio)
    combined = t_alpha * stft_spec + beta * cqt_spec
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
    return combined

def compute_three_channel_spec(audio):
    base = get_weighted_spectrogram(audio)
    delta = librosa.feature.delta(base)
    delta2 = librosa.feature.delta(base, order=2)
    spec_3ch = np.stack([base, delta, delta2], axis=-1)
    return base, delta, delta2, np.expand_dims(spec_3ch, axis=0).astype(np.float32)

def process(file_path):
    audio, _ = librosa.load(file_path, sr=SR)
    if len(audio) < SR * DURATION:
        audio = np.pad(audio, (0, SR * DURATION - len(audio)))
    else:
        audio = audio[:SR * DURATION]
    base, delta, delta2, spec_3ch = compute_three_channel_spec(audio)
    return audio, base, delta, delta2, spec_3ch

def plot_spectrogram(spec, title, sr=SR, hop_length=HOP_LENGTH, y_axis='linear'):
    db_spec = librosa.amplitude_to_db(spec, ref=np.max)
    fig, ax = plt.subplots(figsize=(6, 4))
    img = librosa.display.specshow(
        db_spec,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis=y_axis,
        cmap='magma',
        ax=ax
    )
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Frequency (Hz)", fontsize=12)

    # Add colorbar only if img created
    if img:
        fig.colorbar(img, ax=ax, format="%+2.0f dB", label="Amplitude (dB)")
    st.pyplot(fig)

def plot_waveform(audio, sr):
    fig, ax = plt.subplots(figsize=(6, 3))
    librosa.display.waveshow(audio, sr=sr, ax=ax)
    ax.set_title("Audio Waveform", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Amplitude", fontsize=12)
    st.pyplot(fig)

# --- UI Layout ---

st.title("🔍 Audio Real vs Fake Detector")
st.caption("Upload an audio clip (2 seconds preferred) and detect deepfakes using spectrogram-based features.")

uploaded_file = st.file_uploader("🎧 Upload an audio file", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    audio, base, delta, delta2, processed = process(temp_path)

    prediction = model.predict(processed)[0][0]
    result = "🎯 Real Audio" if prediction > 0.5 else "⚡ Fake Audio"

    st.audio(uploaded_file)
    st.markdown(
        f"<h3 style='color:#1E3A8A; font-weight:bold;'>Prediction: {result}</h3>",
        unsafe_allow_html=True
    )

    st.divider()
    st.subheader("📊 Visualize Audio & Spectrograms")
    if st.button("Show Audio Waveform"):
        plot_waveform(audio, SR)

    st.markdown("### STFT Spectrogram")
    if st.button("Show STFT Spectrogram"):
        stft_spec = compute_stft(audio)
        plot_spectrogram(stft_spec, "STFT Spectrogram", y_axis='linear')


    st.markdown("### CQT Spectrogram")
    if st.button("Show CQT Spectrogram"):
        cqt_spec = compute_cqt(audio)
        plot_spectrogram(cqt_spec, "CQT Spectrogram", y_axis='log')

    st.markdown("### Weighted Spectrogram (STFT + CQT)")
    if st.button("Show Weighted Spectrogram"):
        plot_spectrogram(base, "Weighted Spectrogram", y_axis='linear')
        
    st.markdown("### Δ & ΔΔ Spectrograms")
    if st.button("Show Δ & ΔΔ Spectrograms"):
        st.markdown("**Δ Spectrogram (1st Order)**")
        plot_spectrogram(delta, "Delta Spectrogram", y_axis='linear')
        st.markdown("**ΔΔ Spectrogram (2nd Order)**")
        plot_spectrogram(delta2, "Delta-Delta Spectrogram", y_axis='linear')


    os.remove(temp_path)
