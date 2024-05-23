import os 
import sys
sys.path.insert(0, '/home/george/Desktop/NeMo/nemo/collections/asr/parts/utils')

from audio_utils import get_samples, select_channels
from icecream import ic
import librosa
import soundfile as sf
import numpy as np

audio_file_pzm = "pzm12_segment_3.wav"
audio_file_an4 = "an4_diarize_test.wav"
dtype = 'float32'

# with sf.SoundFile(audio_file_pzm, 'r') as f:
#     samples_pzm = f.read(dtype=dtype)
    # ic(samples_pzm)
    # ic(samples_pzm.shape)
    # ic(samples_pzm.transpose().shape)
    # ic(np.pad(samples_pzm.transpose(), (0, 3)).shape[-1])

# with sf.SoundFile(audio_file_an4, 'r') as f:
#     samples_an4 = f.read(dtype=dtype)
#     ic(samples_an4)
#     ic(samples_an4.shape)
#     ic(samples_an4.transpose().shape)
#     ic(np.pad(samples_an4.transpose(), (0, 3)).shape[-1])

samples_pzm = get_samples(audio_file_pzm)
ic(samples_pzm)
ic(samples_pzm.shape)
if samples_pzm.ndim != 1:
    samples_pzm = samples_pzm[0, :]
ic(samples_pzm.shape)

samples_an4 = get_samples(audio_file_an4)
ic(samples_an4)
ic(samples_an4.shape)
