# Speaker Verification System

A robust web application built with [Streamlit](https://streamlit.io/) that identifies and verifies speakers based on their voice. Powered by a deep learning model, this system extracts audio features (MFCCs) and generates speaker embeddings to enroll and authenticate users in real time. Whether for security or personalization, this app provides a seamless way to register and verify voices.

**Live Demo**: [Try the App](https://speaker-verification.streamlit.app/)

## Features
- User-friendly interface with two tabs: "Enroll Speaker" and "Verify Speaker".
- Enroll speakers by uploading multiple audio samples (FLAC/WAV) and assigning a name.
- Verify speakers by comparing a new audio file against enrolled voices.
- Displays verification results with a similarity score for transparency.
- Sidebar showing all enrolled speakers for easy reference.
- Features a sleek loading animation and processing spinner for a polished experience.

## Repository Structure
```
speaker-verification-system/
├── app.py                 # Streamlit application script
├── requirements.txt       # Project dependencies
├── .gitignore             # Git ignore rules
├── model.pt               # Pre-trained speaker identification model
└── speaker_recognition.py # Module with audio processing and model logic
```

## How It Works
1. Loads a pre-trained speaker identification model (`model.pt`) on startup.
2. **Enrollment**:
   - Accepts multiple audio files (FLAC/WAV) and a speaker name.
   - Extracts MFCC features from each file and computes an average embedding.
   - Stores the embedding in memory under the speaker’s name.
3. **Verification**:
   - Takes a single audio file and compares it to a selected enrolled speaker’s embedding.
   - Uses cosine similarity to determine a match (threshold: 0.5320).
   - Outputs "Match" or "No Match" with a similarity score.

## Model Details
The system uses a deep learning model (`model.pt`) and supporting functions sourced from my repository: [Udemy-Speaker-Recognition](https://github.com/bhavya257/Udemy-Speaker-Recognition/blob/main/speaker_recognition.py).

## Prerequisites
- Python 3.8 or higher
- Git
- Audio files in FLAC or WAV format for testing

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/bhavya257/Speaker-Verification-System.git
   cd speaker-verification-system
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Locally
1. Ensure `app.py`, `model.pt`, and the `speaker_recognition` folder are in the same directory.
2. Start the Streamlit app:
   ```bash
   streamlit run app.py --server.fileWatcherType=none
   ```
3. Open your browser and visit `http://localhost:8501` to interact with the app.

## Deployment
Deploy this app using [Streamlit Community Cloud](https://streamlit.io/cloud):
1. Push the repository to your GitHub account.
2. Log in to Streamlit Community Cloud with GitHub.
3. Create a new app, select your repo, and set `app.py` as the main file.
4. Ensure `model.pt` and the `speaker_recognition` folder are included in the repo root.
5. Deploy and access your app via the provided URL.

## Dependencies
Defined in `requirements.txt`:
- `streamlit` - For the web interface
- `numpy` - For numerical computations
- `torch` - For the deep learning model
- `librosa` - For audio feature extraction (MFCCs)

## Usage Tips
- Use clear, high-quality audio recordings for better accuracy.
- Enroll a speaker with multiple samples to capture voice variations.
- Adjust the similarity threshold (default: 0.5320) in `app.py` if needed for your use case.

## License
This project is licensed under the [MIT License](LICENSE.txt).

## Acknowledgments
- Built using components from the [Udemy-Speaker-Recognition repository](https://github.com/bhavya257/Udemy-Speaker-Recognition).
- Gratitude to the open-source community, especially the teams behind Streamlit, PyTorch, and Librosa.
- Special thanks to [Quan Wang](https://www.linkedin.com/in/wangquan/) for [Speaker Recognition](https://www.udemy.com/course/speaker-recognition/) course on Udemy.