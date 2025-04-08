import streamlit as st
import numpy as np
import time
from speaker_recognition import SpeakerIdentificationModel
from speaker_recognition import extract_mfcc_features, get_embedding, cosine_similarity
from speaker_recognition import AudioConfig, ModelConfig
import os

# Set page configuration
st.set_page_config(
    page_title="Speaker Verification System",
    page_icon="favicon.ico",
)

# Full-page loading screen HTML/CSS with gradient spinner
loading_html = """
<div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: #0E1117; z-index: 9999; display: flex; flex-direction: column; justify-content: center; align-items: center;">
    <div style="color: white; font-size: 24px; text-align: center;">
        <span style="display: inline-block; width: 40px; height: 40px; border-radius: 50%; background: linear-gradient(90deg, rgb(255, 75, 75), rgb(255, 253, 128)); animation: spin 1s linear infinite; position: relative; margin-bottom: 10px;">
            <span style="position: absolute; top: 4px; left: 4px; width: 32px; height: 32px; background: #0E1117; border-radius: 50%;"></span>
        </span>
        <br>Loading Speaker Verification System...
    </div>
</div>
<style>
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
</style>
"""

# Custom spinner for processing
spinner_html = """
<div style="display: flex; justify-content: center; align-items: center; padding: 10px;">
    <div style="width: 30px; height: 30px; border: 4px solid #ddd; border-top: 4px solid #ff5733; border-radius: 50%; animation: spin 1s linear infinite;"></div>
    <span style="margin-left: 10px; font-size: 18px; color: #ff5733;">Processing...</span>
</div>
<style>
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
</style>
"""

# Footer HTML/CSS with gradient link
footer_html = """
<div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 10px 0; text-align: center; z-index: 1000;">
    <p style="margin: 0; font-size: 14px; color: white;">
        Discover more of my work at 
        <a href="https://bhavyajain.in/" target="_blank" style="background: linear-gradient(90deg, rgb(255, 75, 75), rgb(255, 253, 128)); -webkit-background-clip: text; color: transparent; text-decoration: none;">bhavyajain.in</a>.
    </p>
</div>
"""

# Load model on first run
if 'loaded' not in st.session_state:
    st.markdown(loading_html, unsafe_allow_html=True)
    time.sleep(0.5)  # Simulate loading time
    audio_config = AudioConfig()
    model_config = ModelConfig()
    model = SpeakerIdentificationModel(model_config, audio_config, "model.pt")
    st.session_state['loaded'] = True
    st.session_state['model'] = model
    st.session_state['audio_config'] = audio_config
    st.session_state['enrolled_speakers'] = {}  # Dictionary to store enrolled embeddings
    st.rerun()

# Access loaded objects
model = st.session_state['model']
audio_config = st.session_state['audio_config']
enrolled_speakers = st.session_state['enrolled_speakers']

# Sample audio paths
SAMPLE_AUDIO_PATHS = {
    "speaker_1": [
        "sample_audio/speaker_1/speaker1_sample_01.flac",
        "sample_audio/speaker_1/speaker1_sample_02.flac",
        "sample_audio/speaker_1/speaker1_sample_03.flac",
        "sample_audio/speaker_1/speaker1_sample_04.flac",
    ],
    "speaker_2": [
        "sample_audio/speaker_2/speaker2_sample_01.flac",
        "sample_audio/speaker_2/speaker2_sample_02.flac",
        "sample_audio/speaker_2/speaker2_sample_03.flac",
        "sample_audio/speaker_2/speaker2_sample_04.flac",
    ]
}


# Enroll a speaker
def enroll_speaker(speaker_name, audio_files):
    embeddings = []
    for audio_file in audio_files:
        if isinstance(audio_file, str):  # For preloaded file paths
            features = extract_mfcc_features(audio_file, audio_config)
        else:  # For uploaded files
            features = extract_mfcc_features(audio_file.read(), audio_config)
        embedding = get_embedding(features, model)
        embeddings.append(embedding)
    avg_embedding = np.mean(embeddings, axis=0)
    enrolled_speakers[speaker_name] = avg_embedding
    return avg_embedding


# Verify a speaker
def verify_speaker(audio_file, enrolled_embedding, threshold=0.7):
    if isinstance(audio_file, str):  # For preloaded file paths
        features = extract_mfcc_features(audio_file, audio_config)
    else:  # For uploaded files
        features = extract_mfcc_features(audio_file.read(), audio_config)
    test_embedding = get_embedding(features, model)
    similarity = cosine_similarity(test_embedding, enrolled_embedding)
    return similarity >= threshold, similarity


# Streamlit app title
st.title("Speaker Verification System")

# Instructions section
with st.expander("How to Use This App", expanded=False):
    st.markdown("""
    ### Simple Instructions
    1. **Enroll a Speaker**:
       - Go to the "Enroll Speaker" tab.
       - Enter a name or preload sample speakers from the sidebar.
       - Upload audio files or use preloaded samples, then click "Enroll".
    2. **Verify a Speaker**:
       - Go to the "Verify Speaker" tab.
       - Select an enrolled speaker.
       - Upload an audio or select a sample, then click "Verify".
    3. **Sample Audios**:
       - Download or preload sample audio files from the sidebar for testing.
    - **Tips**:
    - Use clear audio files for best results.
    - Enroll multiple samples of a speaker’s voice for better accuracy.
    """)

# Sidebar for sample audio management
st.sidebar.subheader("Sample Audios")
for speaker, files in SAMPLE_AUDIO_PATHS.items():
    with st.sidebar.expander(f"{speaker} Samples"):
        for file_path in files:
            if os.path.exists(file_path):
                file_name = os.path.basename(file_path)
                with open(file_path, "rb") as f:
                    st.download_button(
                        label=f"Download {file_name}",
                        data=f,
                        file_name=file_name,
                        mime="audio/flac"
                    )
            else:
                st.warning(f"File {file_path} not found.")

# Option to preload sample speakers
if st.sidebar.button("Preload Sample Speakers"):
    with st.spinner("Preloading sample speakers..."):
        for speaker, files in SAMPLE_AUDIO_PATHS.items():
            if speaker not in enrolled_speakers and all(os.path.exists(f) for f in files):
                enroll_speaker(speaker, files)
        st.sidebar.success("Sample speakers preloaded!")

# Tabs for enrollment and verification
tab1, tab2 = st.tabs(["Enroll Speaker", "Verify Speaker"])

# Enrollment Tab
with tab1:
    st.subheader("Enroll a New Speaker")
    speaker_name = st.text_input("Speaker Name")
    uploaded_files = st.file_uploader("Upload Audio Files",
                                      type=["wav", "flac", "aiff", "aif", "aifc", "mp3", "m4a", "aac"],
                                      accept_multiple_files=True)
    enroll_placeholder = st.empty()

    if st.button("Enroll"):
        if speaker_name and uploaded_files:
            with enroll_placeholder.container():
                st.markdown(spinner_html, unsafe_allow_html=True)
                enroll_speaker(speaker_name, uploaded_files)
            enroll_placeholder.empty()
            with enroll_placeholder.container():
                st.success(f"Speaker '{speaker_name}' enrolled successfully!")
        else:
            enroll_placeholder.empty()
            with enroll_placeholder.container():
                st.warning("Please provide a speaker name and at least one audio file.")

# Verification Tab
with tab2:
    st.subheader("Verify a Speaker")
    selected_speaker = st.selectbox("Select Enrolled Speaker",
                                    options=list(enrolled_speakers.keys()) if enrolled_speakers else [
                                        "No speakers enrolled"])
    verify_file = st.file_uploader("Upload Audio to Verify",
                                   type=["wav", "flac", "aiff", "aif", "aifc", "mp3", "m4a", "aac"],
                                   accept_multiple_files=False)

    # Option to use sample audio for verification
    sample_verify_file = st.selectbox("Or Use Sample Audio",
                                      ["None"] + [os.path.basename(f) for files in SAMPLE_AUDIO_PATHS.values() for f in
                                                  files])
    verify_placeholder = st.empty()

    if st.button("Verify"):
        if selected_speaker != "No speakers enrolled":
            if verify_file or sample_verify_file != "None":
                with verify_placeholder.container():
                    st.markdown(spinner_html, unsafe_allow_html=True)
                    if sample_verify_file != "None":
                        # Find the full path for the selected sample
                        audio_file = next(f for files in SAMPLE_AUDIO_PATHS.values() for f in files if
                                          os.path.basename(f) == sample_verify_file)
                        if os.path.exists(audio_file):
                            is_match, similarity = verify_speaker(audio_file, enrolled_speakers[selected_speaker],
                                                                  0.5320)
                        else:
                            verify_placeholder.empty()
                            with verify_placeholder.container():
                                st.error(f"Sample file {sample_verify_file} not found.")
                                st.stop()
                    else:
                        is_match, similarity = verify_speaker(verify_file, enrolled_speakers[selected_speaker], 0.5320)
                    time.sleep(0.25)  # Artificial delay for effect
                verify_placeholder.empty()
                with verify_placeholder.container():
                    result = "Match" if is_match else "No Match"
                    st.write(f"Result: **{result}**")
                    st.write(f"Similarity Score: **{similarity:.4f}**")
            else:
                verify_placeholder.empty()
                with verify_placeholder.container():
                    st.warning("Please upload an audio file or select a sample.")
        else:
            verify_placeholder.empty()
            with verify_placeholder.container():
                st.warning("Please select a speaker.")

# Display enrolled speakers
if enrolled_speakers:
    st.sidebar.subheader("Enrolled Speakers")
    for speaker in enrolled_speakers.keys():
        st.sidebar.write(f"- {speaker}")

# Display footer
st.markdown(footer_html, unsafe_allow_html=True)