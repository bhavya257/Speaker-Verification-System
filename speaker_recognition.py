from dataclasses import dataclass

import librosa
import numpy as np
import torch
import torch.nn as nn


@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 16000
    n_mfcc: int = 40
    sequence_length: int = 100  # in frames


@dataclass
class ModelConfig:
    """LSTM model configuration"""
    lstm_layers: int = 3
    hidden_size: int = 64
    batch_size: int = 8
    learning_rate: float = 0.0001


class SpeakerIdentificationModel(nn.Module):
    """LSTM-based speaker identification model"""

    def __init__(self, config: ModelConfig, audio_config: AudioConfig, saved_model: str = ""):
        super().__init__()
        self.config = config
        self.audio_config = audio_config

        self.lstm = nn.LSTM(
            input_size=audio_config.n_mfcc,
            hidden_size=config.hidden_size,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        if saved_model:
            self.load_from_checkpoint(saved_model)

    def load_from_checkpoint(self, checkpoint_path: str):
        """Load model state from checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint["model_state_dict"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model"""
        bidirectional_factor = 2
        h0 = torch.zeros(bidirectional_factor * self.config.lstm_layers,
                         x.shape[0], self.config.hidden_size)
        c0 = torch.zeros(bidirectional_factor * self.config.lstm_layers,
                         x.shape[0], self.config.hidden_size)

        output, _ = self.lstm(x, (h0, c0))
        return torch.mean(output, dim=1)


def extract_mfcc_features(file_path: str, config: AudioConfig) -> np.ndarray:
    """Extract MFCC features from audio file"""
    audio, _ = librosa.load(file_path, sr=config.sample_rate, mono=True)
    mfcc = librosa.feature.mfcc(y=audio, sr=config.sample_rate, n_mfcc=config.n_mfcc)
    return mfcc.transpose()


def get_embedding(features: np.ndarray, model: SpeakerIdentificationModel) -> np.ndarray:
    """Get embedding of an utterance using the model"""
    batch_input = torch.unsqueeze(torch.from_numpy(features), dim=0).float()
    return model(batch_input)[0].detach().numpy()


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings"""
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
