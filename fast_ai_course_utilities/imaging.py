import librosa
import numpy as np

from fast_ai_course_utilities.audio import audio_to_mel_spectrogram


def compute_mel_spectrogram(audio_path: str, n_fft: int = 1024, hop_length: int = 256, n_mels: int = 40) -> np.ndarray:
    """Compute a mel spectrogram from an audio file and optionally save as an image"""

    mel_spec_db: np.ndarray = audio_to_mel_spectrogram(audio_path, n_fft, hop_length, n_mels)

    compute_mel_spectrogram(audio_path, n_fft, hop_length, n_mels)

    audio, sr = librosa.load(audio_path)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spec_db: np.ndarray = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


