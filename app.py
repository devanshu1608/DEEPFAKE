import os
import numpy as np
import librosa
import streamlit as st
import tensorflow as tf

# Audio parameters
SR = 16000              # Sample rate
DURATION = 2            # seconds per segment
N_FFT = 1024            # FFT window size
HOP_LENGTH = 512        # hop length

# Weights for weighted average of STFT and CQT
t_alpha = 0.6          # weight for STFT
beta = 0.4             # weight for CQT

# Spectrogram shape
INPUT_SHAPE = (64, 64, 3)

# Load your trained model
model = tf.keras.models.load_model('retrained_final_model.h5')


# Spectrogram utilities

def compute_stft(audio):
    stft = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH))
    stft = librosa.util.fix_length(stft, size=64, axis=1)
    stft = librosa.util.fix_length(stft, size=64, axis=0)
    return stft


def compute_cqt(audio):
    cqt = np.abs(librosa.cqt(audio, sr=SR, hop_length=HOP_LENGTH))
    cqt = librosa.util.fix_length(cqt, size=64, axis=1)
    cqt = librosa.util.fix_length(cqt, size=64, axis=0)
    return cqt


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
    # fix lengths
    for spec in (base, delta, delta2):
        librosa.util.fix_length(spec, size=64, axis=0)
        librosa.util.fix_length(spec, size=64, axis=1)
    spec_3ch = np.stack([base, delta, delta2], axis=-1)
    return spec_3ch


def process(file_path):
    # Load & trim/pad audio
    audio, _ = librosa.load(file_path, sr=SR)
    if len(audio) < SR * DURATION:
        audio = np.pad(audio, (0, SR * DURATION - len(audio)))
    else:
        audio = audio[:SR * DURATION]

    # Compute 3-channel spectrogram
    spec = compute_three_channel_spec(audio)
    # Add batch dimension and ensure dtype
    spec = np.expand_dims(spec, axis=0).astype(np.float32)
    return spec


# Streamlit UI
st.title("🎵 Audio Real vs Fake Detector")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    # Save temporarily
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # Preprocess and predict
    processed = process(temp_path)
    prediction = model.predict(processed)

    # Interpret result
    result = "🎯 Real Audio" if prediction[0][0] > 0.5 else "⚡ Fake Audio"
    st.success(f"Prediction: {result}")

    # Play audio
    st.audio(uploaded_file)

    # Clean up
    os.remove(temp_path)
