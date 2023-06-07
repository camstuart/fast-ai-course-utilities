import librosa
import numpy as np

from fast_ai_course_utilities.audio import torch_audio_to_mel_spectrogram


def compute_mel_spectrogram(audio_path: str, n_fft: int = 1024, hop_length: int = 256, n_mels: int = 40) -> np.ndarray:
    """Compute a mel spectrogram from an audio file and optionally save as an image"""
    audio, sr = librosa.load(audio_path)
    mel_spec_db: np.ndarray = torch_audio_to_mel_spectrogram(sr, audio, n_fft, hop_length, n_mels)
    return mel_spec_db


