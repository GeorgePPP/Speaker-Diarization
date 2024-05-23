import os
import json
import sys
from icecream import ic

sys.path.insert(0, "/home/george/Desktop/NeMo")

import torch
from omegaconf import OmegaConf
from nemo.collections.asr.parts.utils.decoder_timestamps_utils import ASRDecoderTimeStamps
from nemo.collections.asr.parts.utils.diarization_utils import OfflineDiarWithASR 

# Initialize config
ROOT = "/home/george/Desktop/NeMo/custom"
CONFIG = "/home/george/Desktop/NeMo/examples/speaker_tasks/diarization/conf/inference/diar_infer_meeting.yaml"
AUDIO_NAME = 'IS1009b'
audio_fp = os.path.join(ROOT, AUDIO_NAME + '.wav')
cfg = OmegaConf.load(CONFIG)

# Initialize manifest File
meta = {    
    'audio_filepath': audio_fp, 
    'offset': 0, 
    'duration': None, 
    'label': 'infer', 
    'text': '-', 
    'num_speakers': None, 
    'rttm_filepath': None, 
    'uem_filepath' : None
}

with open(os.path.join(ROOT,'input_manifest.json'),'w') as fp:
    json.dump(meta,fp)
    fp.write('\n')

# Customize general config
pretrained_speaker_model='titanet_small'
cfg.diarizer.manifest_filepath = os.path.join(ROOT,'input_manifest.json')
cfg.diarizer.out_dir = os.path.join(ROOT, 'result') #Directory to store intermediate files and prediction outputs
cfg.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
cfg.diarizer.clustering.parameters.oracle_num_speakers=False

# Using Neural VAD and Conformer ASR 
cfg.diarizer.vad.model_path = 'vad_marblenet'
cfg.diarizer.asr.model_path = 'stt_en_conformer_ctc_large'
cfg.diarizer.oracle_vad = False # ----> Not using oracle VAD 
cfg.diarizer.asr.parameters.asr_based_vad = False

# Run ASR 
asr_decoder_ts = ASRDecoderTimeStamps(cfg.diarizer)
asr_model = asr_decoder_ts.set_asr_model()
word_hyp, word_ts_hyp = asr_decoder_ts.run_ASR(asr_model)
torch.cuda.empty_cache()

print("Decoded word output dictionary: \n", word_hyp[AUDIO_NAME])
print("Word-level timestamps dictionary: \n", word_ts_hyp[AUDIO_NAME])

# Run diarization with extracted word timestamps
asr_diar_offline = OfflineDiarWithASR(cfg.diarizer) # Initialzie diarizer
asr_diar_offline.word_ts_anchor_offset = asr_decoder_ts.word_ts_anchor_offset
diar_hyp, diar_score = asr_diar_offline.run_diarization(cfg, word_ts_hyp)
print("Diarization hypothesis output: \n", diar_hyp[AUDIO_NAME])

# Get speaker-labeled ASR transcription
trans_info_dict = asr_diar_offline.get_transcript_with_speaker_labels(diar_hyp, word_hyp, word_ts_hyp)