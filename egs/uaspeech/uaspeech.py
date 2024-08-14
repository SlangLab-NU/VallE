import logging
from natsort import natsorted
import re
import shutil
import tarfile
import zipfile
import os
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import AlignmentItem, SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, is_module_available, resumable_download, safe_extract

UASPEECH_FULL = ["noisereduced", "normalized", "original"]

UASPEECH_NORMALIZED = "normalized"

METADATA_FILE = "_word.mlf"

UASPEECH_PATH = "/home/data1/data/UASpeech"
log_path = os.getcwd() + "/uaspeechReport.log"

logging.basicConfig(filename=log_path, format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug("This is a debug log")
logger.info("This is an info log")
logger.critical("This is critical")
logger.error("An error occurred")

# UASpeech is not downloadable by url, so we do not need a download method.
# Might not actually need this method. Ordering utterances doesn't matter in the end
# But I could modify this to return the label and sequence to clean up the data prep
# method
def read_mlf(file_path):
    """
    Sorts metadata to match audio files.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    mlf_data = []
    current_entry = None
    b1_count = 0
    b2_count = 0
    b3_count = 0

    for line in lines:
        line = line.strip()

        # Skip comments or empty lines
        if line.startswith("#") or line.startswith(".") or not line:
            continue

        # Start of a new entry
        if line.startswith('"'):
            if current_entry:
                if line.__contains__("_B1_"):
                    b1_count += 1
                elif line.__contains__("_B2_"):
                    b2_count += 1
                elif line.__contains__("_B3_"):
                    b3_count += 1
                mlf_data.append(current_entry)
                # line[3:-5] removes the '*/' and '.lab'
            current_entry = {'label': line[3:-5], 'sequence': ''}
        else:
            # Parse sequence information
            current_entry['sequence']=line
    print(f"b1: {b1_count}")
    print(f"b2: {b2_count}")
    print(f"b3: {b3_count}")
    # Append the last entry
    if current_entry:
        mlf_data.append(current_entry)
    """ Lines will look like
    {'label': 'CF03_B3_UW99_M7', 'sequence': 'AWAY'}
    {'label': 'CF03_B3_UW99_M8', 'sequence': 'AWAY'}
    {'label': 'CF03_B3_UW100_M3', 'sequence': 'CRAYON'}
    {'label': 'CF03_B3_UW100_M5', 'sequence': 'CRAYON'}
    """ 
    mlf_data_sorted = natsorted(mlf_data, key=lambda x: x['label'])
    return mlf_data_sorted


def extract_mlf_information(speaker_dict: dict, speaker_path: str):
    """
    Processes an mlf file and inserts the utterance code
    """
    with open(speaker_path, 'r') as file:
        lines = file.readlines()
        current_entry = None
        for line in lines:
            line = line.strip()
            if line.startswith("#") or line.startswith(".") or not line:
                continue
            if line.startswith('"'):
                current_entry = line[3:-5]
                speaker_dict[current_entry] = None
            else: # get the utterance text
                if current_entry is not None:
                    speaker_dict[current_entry] = line
    return speaker_dict


def standardize_key(key):
    """
    Function to standardize the keys by removing the first character. Need this for finding
    intersection between typical and atypical speaker utterances. Typical speakers will start
    with a 'C'
    """
    return key[1:]


def create_speaker_speaker_pair(
    corpus_dir: Pathlike,
    control_speakers: List[str],
    atypical_speakers: List[str],
    alignments_dir: Optional[Pathlike] = None,
    dataset_parts: Union[str, Sequence[str]] = "auto",
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, RecordingSet]]]:
    corpus_dir = Path(corpus_dir)
    corpus_audio_dir = Path(os.path.join(corpus_dir, "audio"))
    alignments_dir = Path(alignments_dir) if alignments_dir is not None else corpus_audio_dir
    assert corpus_audio_dir.is_dir(), f"No such directory: {corpus_audio_dir}"

    if dataset_parts == "auto":
        dataset_parts = set(UASPEECH_FULL).intersection(path.name for path in corpus_audio_dir.glob("*"))
        if not dataset_parts:
            raise ValueError(f"Could not find any of UASpeech dataset parts in: {corpus_audio_dir}")
    elif isinstance(dataset_parts, str):
        dataset_parts = [dataset_parts]
    logger.info(f"dataset_parts: {dataset_parts}")
    manifests = {}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifests = read_manifests_if_cached(dataset_parts=dataset_parts, output_dir=output_dir)

    metadata_mlf_path = os.path.join(corpus_dir, "mlf")

    typical_recording_train_set = []
    atypical_recording_train_set = []
    typical_recording_test_set = []
    atypical_recording_test_set = []

    with ThreadPoolExecutor(num_jobs) as ex:
        for part in tqdm(dataset_parts, desc="Dataset parts"):
            if not part.startswith("."):  # Avoid hidden files
                logging.info(f"Processing UASpeech subset: {part}")
                if manifests_exist(part=part, output_dir=output_dir):
                    logging.info(f"UASpeech subset: {part} already prepared - skipping.")
                    continue

                for typical_speaker, atypical_speaker in tqdm(zip(control_speakers, atypical_speakers), desc="Preparing Speaker Pair"):
                    typical_speaker_path = os.path.join(metadata_mlf_path, typical_speaker, typical_speaker + METADATA_FILE)
                    atypical_speaker_path = os.path.join(metadata_mlf_path, atypical_speaker, atypical_speaker + METADATA_FILE)

                    assert os.path.isfile(typical_speaker_path), f"No such file: {typical_speaker_path}"
                    assert os.path.isfile(atypical_speaker_path), f"No such file: {atypical_speaker_path}"

                    typical_utterances = extract_mlf_information({}, typical_speaker_path)
                    atypical_utterances = extract_mlf_information({}, atypical_speaker_path)

                    standardize_typical_keys = {standardize_key(key): key for key in typical_utterances}
                    intersection_utterances = set(standardize_typical_keys.keys()).intersection(set(atypical_utterances.keys()))

                    typical_utterances = {standardize_typical_keys[key]: typical_utterances[standardize_typical_keys[key]] for key in intersection_utterances}
                    atypical_utterances = {key: atypical_utterances[key] for key in intersection_utterances}

                    assert len(typical_utterances) == len(atypical_utterances), f"Length Mismatch... Typical: {len(typical_utterances)} and Atypical: {len(atypical_utterances)}"
                    logger.info(f"Length of typical: {len(typical_utterances)}, Length of atypical: {len(atypical_utterances)}")

                    for key, value in tqdm(atypical_utterances.items(), desc="Preparing parallel speakers"):
                        try:
                            atypical_recording_id = value + "_" + key
                            atypical_audio_path = corpus_audio_dir / part / atypical_speaker / f"{key}.wav"
                            atypical_recording = Recording.from_file(atypical_audio_path, atypical_recording_id)
                            if "_B2_" in atypical_recording_id:
                                atypical_recording_test_set.append(atypical_recording)
                            else:
                                atypical_recording_train_set.append(atypical_recording)

                            typical_recording_id = value + "_C" + key
                            typical_audio_path = corpus_audio_dir / part / typical_speaker / f"C{key}.wav"
                            typical_recording = Recording.from_file(typical_audio_path, typical_recording_id)
                            if "_B2_" in typical_recording_id:
                                typical_recording_test_set.append(typical_recording)
                            else:
                                typical_recording_train_set.append(typical_recording)

                        except Exception as err:
                            logger.error(err)

    typical_recording_train_set = RecordingSet.from_recordings(typical_recording_train_set)
    atypical_recording_train_set = RecordingSet.from_recordings(atypical_recording_train_set)

    typical_recording_test_set = RecordingSet.from_recordings(typical_recording_test_set)
    atypical_recording_test_set = RecordingSet.from_recordings(atypical_recording_test_set)

    if output_dir is not None:
        typical_recording_train_set.to_file(output_dir / f"uaspeech_recordings_control_speakers_train.jsonl.gz")
        atypical_recording_train_set.to_file(output_dir / f"uaspeech_recordings_atypical_speakers_train.jsonl.gz")
        typical_recording_test_set.to_file(output_dir / f"uaspeech_recordings_control_speakers_test.jsonl.gz")
        atypical_recording_test_set.to_file(output_dir / f"uaspeech_recordings_atypical_speakers_test.jsonl.gz")

    return {
        "typical_train_recordings": typical_recording_train_set,
        "atypical_train_recordings": atypical_recording_train_set,
        "typical_test_recordings": typical_recording_test_set,
        "atypical_test_recordings": atypical_recording_test_set,
    }

# TEMPORARY TEST SCRIPT
###########################################################################################
# mlf_file_path = "/home/data1/data/UASpeech/mlf/CF04/CF04_word.mlf"

# mlf_data = read_mlf(mlf_file_path)
# # label, sequence = mlf_data
# # Print the result
# for entry in mlf_data:
#     print(entry)
#     print(f"Label: {entry['label']}")
#     print("Sequence:", end=" ")
#     for label in entry['sequence']:
#         print(f"  {label}")
# typical = {}
# PATH = "/home/data1/data/UASpeech/mlf/CF02/CF02_word.mlf"
# extract_mlf_information(typical, PATH)

# Issues with CMO9 and feature extraction
control_speakers = ["CF02", "CF03", "CF04", "CM04", "CM05", "CM06", "CM08", "CM10", "CM12", "CM13"]
atypical_speakers = ["F02", "F03", "F04", "M04", "M05", "M07", "M08", "M10", "M11", "M12"]

create_speaker_speaker_pair(UASPEECH_PATH, control_speakers, atypical_speakers, None, "normalized", output_dir="/home/data1/vall-e.git/VallE/egs/uaspeech/data/manifests")

# sets = create_speaker_speaker_pair(UASPEECH_PATH, "CF02", "F02", None, "normalized", output_dir="/home/data1/vall-e.git/VallE/egs/uaspeech/data/manifests")
# typical_train = sets["typical_train_recordings"]
# atypical_train = sets["atypical_train_recordings"]
# typical_test = sets["typical_test_recordings"]
# atypical_test = sets["atypical_test_recordings"]

# print(f"Typical train length {len(typical_train)}")
# print(f"Atypical train length {len(atypical_train)}")
# print(f"Typical test length {len(typical_test)}")
# print(f"Atypical test length {len(atypical_test)}")


# sets = prepare_uaspeech(UASPEECH_PATH, None, "normalized", output_dir="/home/data1/vall-e.git/VallE/egs/uaspeech/data/manifests")
# train_cerebral = sets["train_supervisions_cerebral"]

# for element in train_cerebral:
#     print(element)
############################################################################################