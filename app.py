import streamlit as st
import numpy as np
import time
from speaker_recognition import SpeakerIdentificationModel, extract_mfcc_features, get_embedding, cosine_similarity, AudioConfig, ModelConfig

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
    <div style="width: 30px; height: 30px; border: 4px solid last forever solid #ddd; border-top: 4px solid #ff5733; border-radius: 50%; animation: spin 1s linear infinite;"></div>
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

# Enroll a speaker
def enroll_speaker(speaker_name, audio_files):
    embeddings = []
    for audio_file in audio_files:
        features = extract_mfcc_features(audio_file, audio_config)
        embedding = get_embedding(features, model)
        embeddings.append(embedding)
    avg_embedding = np.mean(embeddings, axis=0)
    enrolled_speakers[speaker_name] = avg_embedding
    return avg_embedding

# Verify a speaker
def verify_speaker(audio_file, enrolled_embedding, threshold=0.7):
    features = extract_mfcc_features(audio_file, audio_config)
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
       - Enter a name for the speaker.
       - Upload one or more audio files (FLAC or WAV format).
       - Click "Enroll" to register the speaker's voice.

    2. **Verify a Speaker**:
       - Go to the "Verify Speaker" tab.
       - Select a previously enrolled speaker from the dropdown.
       - Upload an audio file to verify.
       - Click "Verify" to check if the voice matches.

    3. **Check Results**:
       - After verification, you'll see a "Match" or "No Match" result along with a similarity score.
       - Enrolled speakers are listed in the sidebar for reference.

    **Tips**:
    - Use clear audio files for best results.
    - Enroll multiple samples of a speaker's voice for better accuracy.
    """)

# Tabs for enrollment and verification
tab1, tab2 = st.tabs(["Enroll Speaker", "Verify Speaker"])

# Enrollment Tab
with tab1:
    st.subheader("Enroll a New Speaker")
    speaker_name = st.text_input("Speaker Name")
    uploaded_files = st.file_uploader("Upload Audio Files (FLAC/WAV)",
                                      type=["flac", "wav"],
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
                                   type=["flac", "wav"],
                                   accept_multiple_files=False)

    verify_placeholder = st.empty()

    if st.button("Verify"):
        if selected_speaker != "No speakers enrolled" and verify_file:
            with verify_placeholder.container():
                st.markdown(spinner_html, unsafe_allow_html=True)
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
                st.warning("Please select a speaker and upload an audio file.")

# Display enrolled speakers
if enrolled_speakers:
    st.sidebar.subheader("Enrolled Speakers")
    for speaker in enrolled_speakers.keys():
        st.sidebar.write(f"- {speaker}")

# Display footer
st.markdown(footer_html, unsafe_allow_html=True)