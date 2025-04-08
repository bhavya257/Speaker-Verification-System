# Speaker Verification System

A robust web application built with [Streamlit](https://streamlit.io/) that identifies and verifies speakers based on their voice. Powered by a deep learning model, this system extracts MFCC features and generates speaker embeddings to enroll and authenticate users in real time. Includes sample audio files for easy testing.

**Live Demo**: [Try the App](https://speaker-verification.streamlit.app/)

## Features
- Intuitive interface with "Enroll Speaker" and "Verify Speaker" tabs.
- Enroll speakers by uploading audio samples or preloading included samples.
- Verify speakers with uploaded or sample audio, showing similarity scores.
- Sidebar with enrolled speakers and downloadable sample audios.
- Sleek loading animations and processing spinners for a polished experience.

## Repository Structure
```
speaker-verification-system/
├── app.py                 # Streamlit application script
├── requirements.txt       # Project dependencies
├── .gitignore             # Git ignore rules
├── model.pt               # Pre-trained speaker identification model
├── packages.txt           # Package requirements
├── speaker_recognition.py # Audio processing and model logic
└── sample_audio           # Sample audio files
```

## How It Works
1. Loads a pre-trained model (`model.pt`) on startup.
2. **Enrollment**:
   - Upload audio files or preload samples for a speaker.
   - Extracts MFCC features and computes an average embedding.
   - Stores embeddings in memory by speaker name.
3. **Verification**:
   - Compares an uploaded or sample audio file to an enrolled speaker’s embedding.
   - Uses cosine similarity (threshold: 0.5320) to output "Match" or "No Match".

## Model Details
The deep learning model (`model.pt`) and functions are sourced from my [Udemy-Speaker-Recognition](https://github.com/bhavya257/Udemy-Speaker-Recognition/blob/main/speaker_recognition.py) repository.

## Prerequisites
- Python 3.8 or higher
- Git
- Audio files

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/bhavya257/Speaker-Verification-System.git
   cd speaker-verification-system
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Locally
1. Ensure `app.py`, `model.pt`, `speaker_recognition.py`, and `sample_audio` are in the root directory.
2. Launch the app:
   ```bash
   streamlit run app.py --server.fileWatcherType=none
   ```
3. Visit `http://localhost:8501` in your browser.

## Deployment
Deploy via [Streamlit Community Cloud](https://streamlit.io/cloud):
1. Push the repository to GitHub.
2. Log in to Streamlit Community Cloud with GitHub.
3. Create a new app, select your repo, and set `app.py` as the main file.
4. Ensure `model.pt`, `speaker_recognition.py`, and `sample_audio` are included.
5. Deploy and access your app via the provided URL.

## Dependencies
Defined in `requirements.txt`:
- `streamlit` - Web interface
- `numpy` - Numerical computations
- `torch` - Deep learning model
- `librosa` - MFCC feature extraction

## Usage Tips
- Use high-quality audio for best results.
- Enroll multiple samples per speaker for improved accuracy.
- Download or preload sample audios from the sidebar to test.
- Adjust the similarity threshold (0.5320) in `app.py` if needed.

## License
This project is licensed under the [MIT License](LICENSE.txt).

## Acknowledgments
- Built using components from [Udemy-Speaker-Recognition](https://github.com/bhavya257/Udemy-Speaker-Recognition).
- Thanks to the open-source community, especially Streamlit, PyTorch, and Librosa teams.
- Special gratitude to [Quan Wang](https://www.linkedin.com/in/wangquan/) for the [Speaker Recognition](https://www.udemy.com/course/speaker-recognition/) Udemy course.