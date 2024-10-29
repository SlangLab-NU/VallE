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
    python3 bin/tokenizer.py \
        --src_dir ./data/manifests --output_dir ./data/tokenized

"""
import argparse
import logging
import os
from pathlib import Path

import torch
import torch.multiprocessing
from icefall.utils import get_executor
from lhotse import CutSet, NumpyHdf5Writer
from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import read_manifests_if_cached
from tqdm.auto import tqdm

from valle.data import (
    AudioTokenConfig,
    AudioTokenExtractor,
    TextTokenizer,
    tokenize_text,
)
from valle.data.fbank import get_fbank_extractor
from valle.utils import SymbolTable

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.multiprocessing.set_sharing_strategy("file_system")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path("data/manifests"),
        help="Path to the manifest files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/tokenized"),
        help="Path to the tokenized files",
    )
    parser.add_argument(
        "--text-extractor",
        type=str,
        default="espeak",
        help="espeak or pypinyin or pypinyin_initials_finals",
    )
    parser.add_argument(
        "--audio-extractor",
        type=str,
        default="Encodec",
        help="Encodec or Fbank",
    )
    parser.add_argument(
        "--dataset-parts",
        type=str,
        default="dev-clean test-clean",
        help="Space separated dataset parts",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="libritts",
        help="prefix of the manifest file",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="jsonl.gz",
        help="suffix of the manifest file",
    )
    parser.add_argument(
        "--batch-duration",
        type=float,
        default=400.0,
        help="The maximum number of audio seconds in a batch."
        "Determines batch size dynamically.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    dataset_parts = args.dataset_parts.replace("--dataset-parts", "").strip()
    if dataset_parts == "all":  # LibriTTS
        dataset_parts = [
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
        ]
    # Manifest names need to include 'uaspeech_recordings_{dataset_part}' or 'uaspeech_supervision_{dataset_part}'
    elif dataset_parts == "uaspeech":
        dataset_parts = [
            "typical_speakers_train",
            "atypical_speakers_train",
            "typical_speakers_test",
            "atypical_speakers_test",
        ]
    else:
        dataset_parts = dataset_parts.replace("-p", "").strip().split(" ")

    assert len(dataset_parts) >= 1

    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=args.src_dir,
        prefix=args.prefix,
        suffix=args.suffix,
        types=["recordings", "supervisions", "cuts"],
    )
    logging.info(f"manifests: {manifests}")
    text_tokenizer = None
    if args.text_extractor:
        text_tokenizer = TextTokenizer(backend=args.text_extractor)

    audio_extractor = None
    if args.audio_extractor:
        if args.audio_extractor == "Encodec":
            audio_extractor = AudioTokenExtractor(AudioTokenConfig())
        else:
            assert args.audio_extractor == "Fbank"
            audio_extractor = get_fbank_extractor()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    unique_symbols = set()
    num_jobs = min(32, os.cpu_count())
    logging.info(f"dataset_parts: {dataset_parts} manifests {len(manifests)}")

    prefix = args.prefix
    if prefix and not prefix.endswith("_"):
        prefix = f"{prefix}_"

    source_train_cuts = {}
    source_test_cuts = {}
    target_train_cuts = {}
    target_test_cuts = {}
    
    def get_target_audio_path(source_audio_path, source_prefix, target_prefix):
        """
        Swaps the prefixes in the source audio path to convert it to the target audio path
        @param source_audio_path: The path to the source audio
        @param source_prefix: The prefix of the source speaker i.e., F02_train
        @param target_prefix: The prefix of the target speaker i.e., CF02_train
        @return: target audio path
        """
        return source_audio_path.replace(source_prefix, target_prefix)

    def get_speaker(prefix):
        """
        Isolates the speaker from the prefix (i.e., isolate F02 from F02_Train)
        @param source_prefix: The prefix of the speaker (i.e., F02_train)
        @return: String representation of the speaker
        """
        return prefix.split("_")[1]
    
    def extract_audio_features(speaker_cuts, storage_path):
        """
        Uses Encoded to extract audio codes for a given speaker
        @param speaker_cuts: the speaker cutset to extract features from
        @param storage_path: Path where encodec features will be stored
        @return: the updated cutset
        """
        with torch.no_grad():
            initial_count = len(speaker_cuts)
            print(f"Initial number of cuts: {initial_count}")
            initial_ids = [cut.id for cut in speaker_cuts]
            if torch.cuda.is_available() and args.audio_extractor == "Encodec":
                speaker_cuts = speaker_cuts.compute_and_store_features_batch(
                    extractor=audio_extractor,
                    storage_path=storage_path,
                    num_workers=num_jobs,
                    batch_duration=args.batch_duration,
                    collate=False,
                    overwrite=True,
                    storage_type=NumpyHdf5Writer,
                )
            else:
                speaker_cuts = speaker_cuts.compute_and_store_features(
                    extractor=audio_extractor,
                    storage_path=storage_path,
                    num_jobs=num_jobs if ex is None else 64,
                    executor=ex,
                    storage_type=NumpyHdf5Writer,
                )
            final_count = len(speaker_cuts)
            final_ids = [cut.id for cut in speaker_cuts]
            print(f"Final number of cuts after feature extraction: {final_count}")
            print(f"Missing cut IDs: {set(initial_ids) - set(final_ids)}")
        return speaker_cuts

    def extract_target_features(tgt):
        """
        Extracts the audio features for the target speaker
        @param tgt: dictionary containing the target utterances and audio file paths.
        @return: returns the cutset of the target with the extracted audio features
        """

        for tgt_partition, tgt_cuts in tgt.items():
            print(f" before processing: {len(tgt_cuts)}")
            tgt_speaker = get_speaker(tgt_partition)

            # AudioTokenizer
            if args.audio_extractor:
                print("Audio extractor called")
                if args.audio_extractor == "Encodec":
                    tgt_storage_path = (
                        f"{args.output_dir}/{args.prefix}_encodec_{tgt_partition}"
                    )
                else:
                    tgt_storage_path = (
                        f"{args.output_dir}/{args.prefix}_fbank_{tgt_partition}"
                    )

                if args.prefix.lower() in ["ljspeech", "aishell", "baker", "uaspeech"]:
                    print("Prefix check")
                    tgt_cuts = tgt_cuts.resample(24000)
                    print(f" After processing: {len(tgt_cuts)}")
                # extract_audio_features(tgt_cuts, tgt_storage_path)
        if tgt_cuts is None or tgt_storage_path is None:
            raise ValueError("The audio extractor settings or the prefix did not match the expected conditions.")
        return extract_audio_features(tgt_cuts, tgt_storage_path)

    def extract_text_phonemes(phoneme_symbols, cut_set):
        """
        Extracts the text and phonemes for a given cutset
        @param cut_set: The cutset to extract text and phonemes of
        @return: no return, but writes k2 text symbols to new file
        """
        # cut_set = CutSet.from_manifests(
        #             recordings=m["recordings"],
        #             supervisions=m["supervisions"],
        #         )
        if args.text_extractor:
            print("TEXT EXTRACTOR RAN")
            if (
                args.prefix == "baker"
                and args.text_extractor == "labeled_pinyin"
            ):
                for c in tqdm(cut_set):
                    phonemes = c.supervisions[0].custom["tokens"]["text"]
                    phoneme_symbols.update(phonemes)
            else:
                for c in tqdm(cut_set):
                    if args.prefix == "ljspeech":
                        text = c.supervisions[0].custom["normalized_text"]
                        text = text.replace("”", '"').replace("“", '"')
                        phonemes = tokenize_text(text_tokenizer, text=text)
                    elif args.prefix == "aishell":
                        phonemes = tokenize_text(
                            text_tokenizer, text=c.supervisions[0].text
                        )
                        c.supervisions[0].custom = {}
                    elif args.prefix == "uaspeech":
                        if c.supervisions[0].text != None:
                            phonemes = tokenize_text(
                                text_tokenizer, text=c.supervisions[0].text
                            )
                            c.supervisions[0].custom = {}
                        else:
                            logging.info(f"Supervision empty: {c}")
                    else:
                        assert args.prefix == "libritts"
                        phonemes = tokenize_text(
                            text_tokenizer, text=c.supervisions[0].text
                        )
                    c.supervisions[0].custom["tokens"] = {"text": phonemes}
                    phoneme_symbols.update(phonemes)
        
        logging.info(f"Writing cutset to: {partition}")
        cuts_filename = f"{partition}.json"
        cut_set.to_file(f"{args.output_dir}/cuts_{cuts_filename}")
        # cut_set.to_file(f"{args.output_dir}/{cuts_filename}")
        # TODO Figure out why phonemes aren't being written to file
        # I think it is happening too early here.
        if args.text_extractor:
            unique_phonemes = SymbolTable()
            for s in sorted(list(phoneme_symbols)):
                print(s)
                unique_phonemes.add(s)
            logging.info(f"{len(phoneme_symbols)} unique phonemes: {phoneme_symbols}")

            unique_phonemes_file = f"{args.output_dir}/unique_text_tokens.k2symbols"
            unique_phonemes.to_file(unique_phonemes_file)
    
    def process_src_tgt_cuts(src, tgt):
        """
        Extracts audio features of the source speaker and pairs it with the features of the target speaker
        and writes it to a json or jsonl.gz file
        @param: Dictionary containing source speaker utterances and audio paths
        @param: Dictionary containing target speaker utterances and audio paths
        """
        
        tgt_cuts = extract_target_features(tgt)
        
        for src_partition, src_cuts in src.items():
            src_speaker = get_speaker(src_partition)

            tgt_speaker = get_speaker(next(iter(tgt)))

            # AudioTokenizer
            if args.audio_extractor:
                if args.audio_extractor == "Encodec":
                    src_storage_path = (
                        f"{args.output_dir}/{args.prefix}_encodec_{src_partition}"
                    )
                else:
                    src_storage_path = (
                        f"{args.output_dir}/{args.prefix}_fbank_{src_partition}"
                    )

                if args.prefix.lower() in ["ljspeech", "aishell", "baker", "uaspeech"]:
                    src_cuts = src_cuts.resample(24000)
                
                 # Extract features for the source cuts
                src_cuts = extract_audio_features(src_cuts, src_storage_path)
                print(f" COMPARE CUTS: {len(tgt_cuts)}, {len(src_cuts)}")

                mismatch = []
                # Assign the computed target features to the source
                for src_cut, tgt_cut in zip(src_cuts, tgt_cuts):
                    temp_src = src_cut.id
                    temp_tgt = tgt_cut.id
                    
                    temp_src = temp_src.replace(get_speaker(temp_src), '')
                    temp_tgt = temp_tgt.replace(get_speaker(temp_tgt), '')
                    
                    if temp_src != temp_tgt:
                        mismatch.append(temp_src)
                        continue
                    else:
                        src_cut.target_recording = tgt_cut
            if args.text_extractor:
                print("TEXT EXTRACTOR RAN")
                if (
                    args.prefix == "baker"
                    and args.text_extractor == "labeled_pinyin"
                ):
                    for c in tqdm(src_cuts):
                        phonemes = c.supervisions[0].custom["tokens"]["text"]
                        unique_symbols.update(phonemes)
                else:
                    for c in tqdm(src_cuts):
                        if args.prefix == "ljspeech":
                            text = c.supervisions[0].custom["normalized_text"]
                            text = text.replace("”", '"').replace("“", '"')
                            phonemes = tokenize_text(text_tokenizer, text=text)
                        elif args.prefix == "aishell":
                            phonemes = tokenize_text(
                                text_tokenizer, text=c.supervisions[0].text
                            )
                            c.supervisions[0].custom = {}
                        elif args.prefix == "uaspeech":
                            if c.supervisions[0].text != None:
                                phonemes = tokenize_text(
                                    text_tokenizer, text=c.supervisions[0].text
                                )
                                c.supervisions[0].custom = {}
                            else:
                                logging.info(f"Supervision empty: {c}")
                        else:
                            assert args.prefix == "libritts"
                            phonemes = tokenize_text(
                                text_tokenizer, text=c.supervisions[0].text
                            )
                        c.supervisions[0].custom["tokens"] = {"text": phonemes}
                        unique_symbols.update(phonemes)
        
            logging.info(f"Writing cutset to: {src_partition}")
            cuts_filename = f"{src_partition}.{args.suffix}"
            src_cuts.to_file(f"{args.output_dir}/cuts_{cuts_filename}")
        # cut_set.to_file(f"{args.output_dir}/{cuts_filename}")
        # TODO Figure out why phonemes aren't being written to file
        # I think it is happening too early here.
        if args.text_extractor:
            unique_phonemes = SymbolTable()
            for s in sorted(list(unique_symbols)):
                print(s)
                unique_phonemes.add(s)
            logging.info(f"{len(unique_symbols)} unique phonemes: {unique_symbols}")

            unique_phonemes_file = f"{args.output_dir}/unique_text_tokens.k2symbols"
            unique_phonemes.to_file(unique_phonemes_file)
            # extract_text_phonemes(unique_symbols, src_cuts)
        # print(f"Writing file cuts_{src_partition}.jsonl.gz")
        # src_cuts.to_file(f"{args.output_dir}/cuts_{src_partition}.json")
    

    with get_executor() as ex:
        for partition, m in manifests.items():
            # Need to isolate which partition is typical vs atypical 
            logging.info(
                f"Processing partition: {partition} CUDA: {torch.cuda.is_available()}"
            )
            try:
                cut_set = CutSet.from_manifests(
                    recordings=m["recordings"],
                    supervisions=m["supervisions"],
                )
                if "train" in partition:
                    if "atypical" in partition:
                        source_train_cuts[partition] = cut_set                        
                    else:
                        target_train_cuts[partition] = cut_set
                elif "test" in partition:
                    if "atypical" in partition:
                        source_test_cuts[partition] = cut_set                      
                    else:
                        target_test_cuts[partition] = cut_set
     
                    # cut.target_recording = Recording.from_file
            except Exception:
                cut_set = m["cuts"]
        process_src_tgt_cuts(source_train_cuts, target_train_cuts)
        process_src_tgt_cuts(source_test_cuts, target_test_cuts)

        

if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
