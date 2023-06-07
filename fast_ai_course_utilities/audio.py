from typing import Tuple

import librosa
import numpy as np
import torchaudio
import torch
from pydub import AudioSegment


def scale_minmax(x: np.ndarray, min_value: float = 0.0, max_value: float = 1.0) -> np.ndarray:
    """Rescale an array to be between min_value and max_value"""
    x_std = (x - x.min()) / (x.max() - x.min())
    x_scaled: np.ndarray = x_std * (max_value - min_value) + min_value
    return x_scaled


def load_audio_file(path: str) -> Tuple[np.ndarray, int]:
    audio, sr = librosa.load(path)
    return audio, sr


def librosa_audio_to_mel_spectrogram(sample_rate, audio, n_fft=1024, hop_length=256, n_mels=40) -> np.ndarray:
    """Compute a mel spectrogram from a audio data with librosa"""
    audio = audio.astype(np.float32)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length,
                                              n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    img = scale_minmax(mel_spec_db, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
    return img


def torch_audio_to_mel_spectrogram(sample_rate: int, signal: torch.Tensor, n_fft: int = 1024, hop_length: int = 256,
                                   n_mels: int = 40) -> torch.Tensor:
    """Compute a mel spectrogram from a audio data with pytorch"""
    ms = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    mel_spectrogram: torch.Tensor = ms(signal)
    return mel_spectrogram


def load_torch_audio_file(path: str) -> Tuple[torch.Tensor, int]:
    audio, sr = torchaudio.load(path)
    return audio, sr


def cut_if_necessary(num_samples: int, signal: torch.Tensor) -> torch.Tensor:
    if signal.shape[1] > num_samples:
        signal = signal[:, :num_samples]
    return signal


def right_pad_if_necessary(num_samples: int, signal: torch.Tensor) -> torch.Tensor:
    length_signal = signal.shape[1]
    if length_signal < num_samples:
        num_missing_samples = num_samples - length_signal
        last_dim_padding = (0, num_missing_samples)
        signal = torch.nn.functional.pad(signal, last_dim_padding)
    return signal


def resample_if_necessary(target_sample_rate: int, signal: torch.Tensor, sr: int) -> torch.Tensor:
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        signal = resampler(signal)
    return signal


def mix_down_if_necessary(signal: torch.Tensor) -> torch.Tensor:
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal
