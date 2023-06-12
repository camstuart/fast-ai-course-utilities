import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List

import numpy as np

from fast_ai_course_utilities.audio import audio_to_mfcc_image, file_to_audio


class Classification(Enum):
    human = auto()
    noise = auto()
    testtone = auto()


@dataclass(slots=True)
class AudioTrainingRecord:
    classification: Classification
    # audio_data: np.ndarray
    audio_file: str
    mel_spectrogram: np.ndarray


@dataclass
class AudioTrainingRecords:
    records: List[AudioTrainingRecord] = field(init=False, default_factory=list)

    def load_from_dir(self, base_path: str, ext: str = '.mp3') -> None:
        for parent_dir in os.scandir(base_path):
            if parent_dir.is_dir():
                for audio_file in os.scandir(f'{base_path}/{parent_dir.name}'):
                    if str(audio_file.name.lower()).endswith(ext.lower()):
                        input_audio_file = f'{base_path}/{parent_dir.name}/{audio_file.name}'
                        data = file_to_audio(input_audio_file)
                        mel_spec = audio_to_mfcc_image(audio=data[0], sample_rate=data[1], length=4)

                        if parent_dir.name in Classification.__members__:
                            self.records.append(AudioTrainingRecord(classification=Classification[parent_dir.name],
                                                                    audio_file=input_audio_file,
                                                                    mel_spectrogram=mel_spec))
                        else:
                            print(f'Unknown classification (parent directory): {parent_dir.name}')

    def summary(self) -> None:
        image_byte_count = 0
        for classification in Classification:
            items = 0
            for record in self.records:
                if record.classification == classification:
                    items += 1
                    image_byte_count += record.mel_spectrogram.nbytes

            print(f'classification: "{classification.name}" has: {items} entries')
        print(f'total files   : {len(self.records)}\n'
              # f'audio data    : {(audio_byte_count / 1048576):.2f} MB\n'
              f'images data   : {(image_byte_count / 1048576):.2f} MB')


def get_image_files(data_set: AudioTrainingRecords) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    for record in data_set.records:
        images.append(record.mel_spectrogram)
    return images


def get_parent_label(data_set: AudioTrainingRecords, mel_spec: np.ndarray) -> str:
    label: str = ''
    for record in data_set.records:
        if np.array_equal(record.mel_spectrogram, mel_spec):
            label = record.classification.name
            break
    return label
