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
import os

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
    count = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(delimiter)
            # Ensure the line has at least four parts; fill missing parts with None
            while len(parts) < 4:
                parts.append(None)
            source_transcript, source_path, target_transcript, target_path = parts[:4]
            if not (os.path.isfile(source_path) and os.path.isfile(target_path)):
                print(f"Skipping line due to missing file:\n  source: {source_path}\n  target: {target_path}")
                count += 1
                continue
            source_transcripts.append(source_transcript)
            source_paths.append(source_path)
            target_transcripts.append(target_transcript)
            target_paths.append(target_path)
        print(f"Unable to produce samples for {count} utterances")
    return source_transcripts, source_paths, target_transcripts, target_paths

if __name__=="__main__":
    # Example usage
    parser = argparse.ArgumentParser()
    parser.add_argument("--atyp_to_atyp", type=str, default= '../egs/uaspeech/atyp_to_atyp_inference_list.txt')
    parser.add_argument("--atyp_to_typ", type=str, default= '../egs/uaspeech/atyp_to_typ_inference_list.txt')
    parser.add_argument("--typical-only-mode", type=int, choices=[0,1], help="Run evaluation assuming input is typical speaker inference (typical-to-typical).")
    args = parser.parse_args()
    
    config = GlobalConfig()
    pitch_algorithm = 'yin'  # or 'dio'

    if args.typical_only_mode == 1:
        # Just use one file for both GT and synth paths
        src_txts, gt_paths, _, synth_paths = process_file(args.atyp_to_atyp)
        x_gt, _ = load_audio_paths(gt_paths)
        x_synth, _ = load_audio_paths(synth_paths)
        gt_tensor = [torch.Tensor(x) for x in x_gt]
        synth_tensor = [torch.Tensor(x) for x in x_synth]

        pitch_gt = [torch.Tensor(eval(pitch_algorithm)(x, config)['pitches']).unsqueeze(1) for x in x_gt]
        pitch_synth = [torch.Tensor(eval(pitch_algorithm)(x, config)['pitches']).unsqueeze(1) for x in x_synth]

        # Filter valid
        valid = [
            (gt, synth, gt_p, synth_p)
            for gt, synth, gt_p, synth_p in zip(gt_tensor, synth_tensor, pitch_gt, pitch_synth)
            if gt.shape[0] > 0 and synth.shape[0] > 0 and gt_p.shape[0] > 0 and synth_p.shape[0] > 0
        ]
        if not valid:
            raise ValueError("No valid pairs found for evaluation.")
        gt_tensor, synth_tensor, pitch_gt, pitch_synth = zip(*valid)

        # Run metrics
        DTW = {v: k for v, k in enumerate(batch_dynamic_time_warping(pitch_gt, pitch_synth, config.dist_fn, config.norm_align_type)['norm_align_costs'])}
        MCD = {v: k for v, k in enumerate(batch_mel_cepstral_distortion(gt_tensor, synth_tensor, config))}
        WER, CER, hyp_sent = calculate_batched_wer(src_txts, synth_tensor)
        SECS = calculate_speaker_similarity(gt_tensor, synth_tensor)

        print("==== Evaluation (Typical-to-Typical) ====")
        print(f"DTW mean ± std: {np.mean(list(DTW.values())):.3f} ± {np.std(list(DTW.values())):.3f}")
        print(f"MCD mean ± std: {np.mean(list(MCD.values())):.3f} ± {np.std(list(MCD.values())):.3f}")
        print(f"CER: {CER:.3f}")
        print(f"SECS: {SECS:.3f}")
        print(f"HYP Sentences: {hyp_sent}")
    else:
        # Load atyp_to_atyp for WER/CER/SECS
        atyp_src_txt, atyp_src_paths, _, synth_paths = process_file(args.atyp_to_atyp)
        x_synths, _ = load_audio_paths(synth_paths)
        x_atyp_gts, _ = load_audio_paths(atyp_src_paths)
        synths_tensor = [torch.Tensor(x) for x in x_synths]
        atyp_gts_tensor = [torch.Tensor(x) for x in x_atyp_gts]

        # Load atyp_to_typ for DTW/MCD/MSD
        _, _, _, typ_target_paths = process_file(args.atyp_to_typ)
        x_typ_gts, _ = load_audio_paths(typ_target_paths)
        typ_gts_tensor = [torch.Tensor(x) for x in x_typ_gts]

        typ_gts_pitch = [torch.Tensor(eval(pitch_algorithm)(x, config)['pitches']).unsqueeze(1) for x in x_typ_gts]
        synths_pitch = [torch.Tensor(eval(pitch_algorithm)(x, config)['pitches']).unsqueeze(1) for x in x_synths]

        # Filter invalid pairs. Some models may produce silent output
        valid = [
            (typ, synth, typ_pitch, synth_pitch)
            for typ, synth, typ_pitch, synth_pitch in zip(typ_gts_tensor, synths_tensor, typ_gts_pitch, synths_pitch)
            if typ.shape[0] > 0 and synth.shape[0] > 0 and typ_pitch.shape[0] > 0 and synth_pitch.shape[0] > 0
        ]
        if not valid:
            raise ValueError("No valid pairs found for DTW/MCD/MSD.")
        
        typ_gts_tensor, synths_tensor, typ_gts_pitch, synths_pitch = zip(*valid)

        DTW = {v: k for v, k in enumerate(batch_dynamic_time_warping(typ_gts_pitch, synths_pitch, config.dist_fn, config.norm_align_type)['norm_align_costs'])}
        MSD = {v: k for v, k in enumerate(batch_mel_spectral_distortion(typ_gts_tensor, synths_tensor, config))}
        MCD = {v: k for v, k in enumerate(batch_mel_cepstral_distortion(typ_gts_tensor, synths_tensor, config))}

        SECS = calculate_speaker_similarity(atyp_gts_tensor, synths_tensor)  # <- always atypical based
        WER, CER, hyp_sent = calculate_batched_wer(atyp_src_txt, synths_tensor)

        print("==== Evaluation Results ====")
        print(f"DTW mean ± std: {np.mean(list(DTW.values())):.3f} ± {np.std(list(DTW.values())):.3f}")
        print(f"MCD mean ± std: {np.mean(list(MCD.values())):.3f} ± {np.std(list(MCD.values())):.3f}")
        # print(f"MSD mean ± std: {np.mean(list(MSD.values())):.3f} ± {np.std(list(MSD.values())):.3f}")

        # print(f"WER: {WER:.3f}")
        print(f"CER: {CER:.3f}")
        print(f"SECS: {SECS:.3f}")
        print(f"HYP Sentences: {hyp_sent}")

    # (source_transcripts,
    #  source_paths,
    #  target_transcripts,
    #  target_paths) = process_file(file_path, delimiter=" ")  # Use "\t" for tab or " " for space
    # #outputs is a tuple source_transcript, source_path, target_transcript, target_path
    # x_synths, _ = load_audio_paths(source_paths)
    # x_gts, _ = load_audio_paths(target_paths)
    # gts_tensor = [torch.Tensor(item) for item in x_gts]
    # synths_tensor = [torch.Tensor(item) for item in x_synths]
    # # Define Configurations
    # config = GlobalConfig()

    # pitch_algorithm = 'yin' #['dio','yin']
    # # Compute Pitch
    # gts_pitch = [eval(pitch_algorithm)(item, config) for item in x_gts]
    # synths_pitch = [eval(pitch_algorithm)(item, config) for item in x_synths]
    # gts_pitch_tensor = [torch.Tensor(item['pitches']).unsqueeze_(1) for item in gts_pitch]
    # synths_pitch_tensor = [torch.Tensor(item['pitches']).unsqueeze_(1) for item in synths_pitch]
    # # Batched Metrics
    # try:
    #     DTW = {v: k for v, k in enumerate(
    #         batch_dynamic_time_warping(gts_pitch_tensor, synths_pitch_tensor, config.dist_fn, config.norm_align_type)[
    #             'norm_align_costs'])}
    #     MSD = {v: k for v, k in enumerate(batch_mel_spectral_distortion(gts_tensor, synths_tensor, config))}
    #     MCD = {v: k for v, k in enumerate(batch_mel_cepstral_distortion(gts_tensor, synths_tensor, config))}        
    #     SECS = calculate_speaker_similarity(gts_tensor, synths_tensor)

    #     print('DTW:', DTW) # compare with typical speaker
    #     print('MCD:', MCD) # compare with typical speaker
    #     print('MSD:', MSD) # compare with typical speaker
    #     print('SECS:', SECS) # compare with atypical speaker
    # except:
    #     print("Skipping DTW, MSD, MCD, SECS due to poor inference generation")
    
    # WER, CER, hyp_sent = calculate_batched_wer(target_transcripts, synths_tensor)      
    # print('WER:', WER) # compare with atypical speaker
    # print('CER:', CER) # compare with atypical speaker
    # print('HYPS:', hyp_sent) # compare with atypical speaker
    