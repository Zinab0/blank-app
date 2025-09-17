import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import tempfile

# -----------------------
# Load trained Autoencoder
# -----------------------

autoencoder = tf.keras.models.load_model("model.h5", compile=False)


# -----------------------
# Preprocessing Function (same as training)
# -----------------------
def extract_logmel_frames(file_path, sr=16000, n_mels=128, hop_length=512):
    y, _ = librosa.load(file_path, sr=sr)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=1024,
        hop_length=hop_length,
        n_mels=n_mels
    )
    logmel = librosa.power_to_db(mel)
    return logmel.T  # shape: (time_steps, n_mels)

# -----------------------
# Reconstruction error function
# -----------------------
def reconstruction_errors(model, data):
    preds = model.predict(data, batch_size=64, verbose=0)
    return np.mean(np.square(data - preds), axis=1)

# -----------------------
# Streamlit UI
# -----------------------
st.title("Anomalous Sound Detection Demo")
st.write("Upload a machine sound file to check if it’s normal or anomalous.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Show audio player
    st.audio(tmp_path)

    # Extract features
    frames = extract_logmel_frames(tmp_path)

    if frames is None or len(frames) == 0:
        st.warning("Could not extract features from this file.")
    else:
        frames = np.atleast_2d(frames)

        # Compute reconstruction error
        errors = reconstruction_errors(autoencoder, frames)
        avg_error = np.mean(errors)

        # Choose threshold (example: 95th percentile of training errors)
        # 👉 adjust based on your validation set
        threshold = 0.05  

        # Display results
        st.subheader("Prediction Result")
        if avg_error > threshold:
            st.error(f"🚨 Anomaly Detected! Reconstruction Error = {avg_error:.4f}")
        else:
            st.success(f"✅ Normal Sound. Reconstruction Error = {avg_error:.4f}")
