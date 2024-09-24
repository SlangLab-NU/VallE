#!/usr/bin/env python3
# Copyright    2023                            (authors: Feiteng Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Phonemize Text and EnCodec Audio.

Usage example:
    python3 bin/infer.py \
        --decoder-dim 128 --nhead 4 --num-decoder-layers 4 --model-name valle \
        --text-prompts "Go to her." \
        --audio-prompts ./prompts/61_70970_000007_000001.wav \
        --output-dir infer/demo_valle_epoch20 \
        --checkpoint exp/valle_nano_v2/epoch-20.pt

"""
import argparse
import logging
import os
from pathlib import Path

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch
import torchaudio
from icefall.utils import AttributeDict, str2bool

from valle.data import (
    AudioTokenizer,
    TextTokenizer,
    tokenize_audio,
    tokenize_text,
)
from valle.data.collation import get_text_token_collater
from valle.models import get_model


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--atypical-audio",
        type=str,
        default="",
        help="Atypical audio prompts which are separated by |.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="exp/vallf_nano_full/checkpoint-100000.pt",
        help="Path to the saved checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("infer/demo"),
        help="Path to the tokenized files.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=-100,
        help="Whether AR Decoder do top_k(if > 0) sampling.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The temperature of AR Decoder top_k sampling.",
    )

    return parser.parse_args()


def load_model(checkpoint, device):
    if not checkpoint:
        return None

    checkpoint = torch.load(checkpoint, map_location=device)

    args = AttributeDict(checkpoint)
    model = get_model(args)

    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model"], strict=True
    )
    assert not missing_keys
    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    audio_tokenizer = AudioTokenizer()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    atypical_audio_prompts = []
    if args.atypical_audio:
        for n, audio_file in enumerate(args.atypical_audio.split(",")):
            encoded_frames = tokenize_audio(audio_tokenizer, audio_file)
            atypical_audio_prompts.append(encoded_frames[0][0])
            print(f"Atypical: {encoded_frames[0][0].shape}")
        atypical_audio_prompts = torch.concat(atypical_audio_prompts, dim=-1).transpose(2, 1)
        atypical_audio_prompts = atypical_audio_prompts.to(device)

    # Synthesize typical speech from atypical speech
    encoded_frames = model.inference(
        atypical_audio_prompts,
        top_k=args.top_k,
        temperature=args.temperature,
    )
    print(f"encoded_frames size: {encoded_frames.shape}")
    samples = audio_tokenizer.decode([(encoded_frames.transpose(2, 1), None)])
    # Save the output audio
    for audio_file in args.atypical_audio.split(","):
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        output_file = f"{args.output_dir}/{base_name}_output_typical_test2.wav"
    torchaudio.save(output_file, samples[0].cpu(), 24000)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)
if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
