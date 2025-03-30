import logging
import warnings

import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from scipy.spatial.distance import cdist


# Basic Imports
import argparse
import json
import numpy as np
import torch

# Import Audio Loader
from audio.helpers import load_audio_paths

# Import Pitch Computation
from audio.pitch import dio, yin

# Import Config
from config.global_config import GlobalConfig

# Import Metrics
from metrics.VDE import voicing_decision_error
from metrics.GPE import gross_pitch_error
from metrics.FFE import f0_frame_error
from metrics.DTW import batch_dynamic_time_warping
from metrics.MSD import batch_mel_spectral_distortion
from metrics.MCD import batch_mel_cepstral_distortion
from metrics.WER import calculate_batched_wer
from metrics.moments import estimate_moments
from metrics.SECS import calculate_speaker_similarity
# Import Basic Stats
#from metrics.helpers import add_basic_stat

def process_file(file_path, delimiter="\t"):
    source_transcripts = []
    source_paths = []
    target_transcripts = []
    target_paths = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(delimiter)
            # Ensure the line has at least four parts; fill missing parts with None
            while len(parts) < 4:
                parts.append(None)
            source_transcript, source_path, target_transcript, target_path = parts[:4]
            source_transcripts.append(source_transcript)
            source_paths.append(source_path)
            target_transcripts.append(target_transcript)
            target_paths.append(target_path)
    return source_transcripts, source_paths, target_transcripts, target_paths

if __name__=="__main__":
    # Example usage
    file_path = "driver.txt"  # Replace with your file path
    (source_transcripts,
     source_paths,
     target_transcripts,
     target_paths) = process_file(file_path, delimiter=" ")  # Use "\t" for tab or " " for space
    #outputs is a tuple source_transcript, source_path, target_transcript, target_path
    x_synths, _ = load_audio_paths(source_paths)
    x_gts, _ = load_audio_paths(target_paths)
    gts_tensor = [torch.Tensor(item) for item in x_gts]
    synths_tensor = [torch.Tensor(item) for item in x_synths]
    # Define Configurations
    config = GlobalConfig()

    pitch_algorithm = 'yin' #['dio','yin']
    # Compute Pitch
    gts_pitch = [eval(pitch_algorithm)(item, config) for item in x_gts]
    synths_pitch = [eval(pitch_algorithm)(item, config) for item in x_synths]
    gts_pitch_tensor = [torch.Tensor(item['pitches']).unsqueeze_(1) for item in gts_pitch]
    synths_pitch_tensor = [torch.Tensor(item['pitches']).unsqueeze_(1) for item in synths_pitch]
    # Batched Metrics
    DTW = {v: k for v, k in enumerate(
        batch_dynamic_time_warping(gts_pitch_tensor, synths_pitch_tensor, config.dist_fn, config.norm_align_type)[
            'norm_align_costs'])}
    MSD = {v: k for v, k in enumerate(batch_mel_spectral_distortion(gts_tensor, synths_tensor, config))}
    MCD = {v: k for v, k in enumerate(batch_mel_cepstral_distortion(gts_tensor, synths_tensor, config))}
    WER, CER, hyp_sent = calculate_batched_wer(target_transcripts, synths_tensor)
    SECS = calculate_speaker_similarity(gts_tensor, synths_tensor)
    print('DTW:', DTW)
    print('MCD:', MCD)
    print('MSD:', MSD)
    print('WER:', WER)
    print('CER:', CER)
    print('HYPS:', hyp_sent)
    print('SECS:', SECS)