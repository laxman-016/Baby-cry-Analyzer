import streamlit as st
import librosa
import numpy as np
import pickle
import soundfile as sf
import time
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
from av import AudioFrame
import base64

# Set Streamlit page configuration
st.set_page_config(page_title="CryML Analyzer", layout="centered")

# Function to encode an image to base64
def get_base64_img(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load background image
bg_img = get_base64_img("back.jpg")

st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bg_img}"); 
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        background-attachment: fixed;
    }}
    .main {{
        background-color: rgba(255, 255, 255, 0.85);
        padding: 20px;
        border-radius: 10px;
    }}
    .title-text {{
        text-align: center;
        font-size: 40px;
        color: black;
        font-weight: bold;
        margin-bottom: 10px;
    }}
    .sub-text {{
        text-align: center;
        font-size: 18px;
        color: black;  /* Changed from blue to black */
        margin-bottom: 20px;
    }}
    .highlight-blue {{
        color: black;  /* Changed from blue to black */
        font-size: 18px;
        font-weight: 500;
        margin-top: 10px;
        margin-bottom: 10px;
    }}
    .webrtc-container {{
        margin-top: 20px;
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 10px;
        border: 2px solid #ccc;
        border-radius: 10px;
    }}
    .button-row {{
        display: flex;
        justify-content: space-evenly;
        align-items: center;
        width: 100%;
    }}
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        with open("xgb_model (1).pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Class labels
class_labels = {
    0: 'tired',
    1: 'hungry',
    2: 'belly_pain',
    3: 'burping',
    4: 'discomfort'
}

# Feature extraction
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40,
                                            n_fft=1024, hop_length=160,
                                            win_length=400, window='hann').T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024,
                                                     hop_length=160, win_length=400,
                                                     window='hann', n_mels=128).T, axis=0)
        stft = np.abs(librosa.stft(y, n_fft=1024, hop_length=160,
                                   win_length=400, window='hann'))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, y=y, sr=sr,
                                                     n_chroma=12).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, y=y, sr=sr,
                                                             n_fft=1024, hop_length=160,
                                                             win_length=400, n_bands=7,
                                                             fmin=100).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr,
                                                  bins_per_octave=12).T, axis=0)
        features = np.concatenate((mfcc, mel, chroma, contrast, tonnetz))
        if features.shape[0] == 193:
            features = np.append(features, 0.0)
        return features
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# Audio processor
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.file_path = "recorded_audio.wav"
        self.audio_data = []
        self.processing = False

    def recv_queued(self, frames: list[AudioFrame]):
        for frame in frames:
            self.audio_data.append(frame.to_ndarray())

    def stop(self):
        if self.audio_data and not self.processing:
            self.processing = True
            audio_np = np.concatenate(self.audio_data, axis=0).flatten()
            sf.write(self.file_path, audio_np, 16000)
            self.audio_data = []

            processing_placeholder = st.empty()
            processing_placeholder.info("🔄 Processing and classifying the recorded audio... Please wait.")
            time.sleep(2)

            features = extract_features(self.file_path)
            if features is not None:
                if features.shape[0] == 194:
                    prediction = model.predict(features.reshape(1, -1))[0]
                    label = class_labels.get(prediction, "Unknown")
                    processing_placeholder.success(f"✅ Predicted Classification: **{label}**")
                else:
                    processing_placeholder.error(f"❌ Feature shape mismatch: {features.shape[0]}")
            else:
                processing_placeholder.error("❌ Feature extraction failed.")
            self.processing = False

# Heading and Subtitle
st.markdown("<div class='title-text'>CryML Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>Detect and analyze your baby’s emotions from their cry 🎧</div>", unsafe_allow_html=True)

# Recording instruction
st.markdown("<div class='highlight-blue'>🎙️ Record an audio to classify the baby's cry. The audio will be stored and processed when you stop recording.</div>", unsafe_allow_html=True)

# WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=AudioProcessor,
    video_processor_factory=None,
)

# Add Start and Stop buttons in the same row
col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ Start Recording"):
        st.write("Recording started...")

with col2:
    if st.button("🛑 Stop Recording"):
        if webrtc_ctx.audio_processor is not None:
            webrtc_ctx.audio_processor.stop()

# Upload instruction
st.markdown("<div class='highlight-blue'>📁 Upload an audio file (wav/mp3)</div>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("", type=["wav", "mp3"])
if uploaded_file is not None:
    file_path = "uploaded_audio.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(file_path, format="audio/wav")

    processing_placeholder_upload = st.empty()
    processing_placeholder_upload.info("🔄 Processing uploaded audio for classification...")
    time.sleep(2)

    features = extract_features(file_path)
    if features is not None:
        if features.shape[0] == 194:
            prediction = model.predict(features.reshape(1, -1))[0]
            label = class_labels.get(prediction, "Unknown")
            processing_placeholder_upload.success(f"✅ Predicted Classification: **{label}**")
        else:
            processing_placeholder_upload.error(f"❌ Feature shape mismatch: {features.shape[0]}")
    else:
        processing_placeholder_upload.error("❌ Error extracting features.")
