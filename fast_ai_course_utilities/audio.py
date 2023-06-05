import librosa
import numpy as np

from fast_ai_course_utilities.data_structures import scale_minmax


def load_audio_file(path: str) -> (np.ndarray, int):
    audio, sr = librosa.load(path)
    return audio, sr


def audio_to_mel_spectrogram(sample_rate, audio, n_fft=1024, hop_length=256, n_mels=40) -> np.ndarray:
    """Compute a mel spectrogram from a audio data"""
    audio = audio.astype(np.float32)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length,
                                              n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    img = scale_minmax(mel_spec_db, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
    return img
