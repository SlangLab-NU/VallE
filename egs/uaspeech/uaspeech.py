import logging
import math, random, re
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


def generate_test_dev_utterances(codes={'D': 10,'L': 26,'C': 19, 'CW': 100,'UW': 100}):
    """
    Generates random numbers which will associate with the UASpeech utterance codes for the categories:
    {Digits, Letters, Computer Commands, Common Words, Uncommon Words}. With an 80/10/10 split the category
    breakdowns are as follows: 
      *  Digits, 8 words train, 1 word val, 1 word test
      *  Letters, 20 words train, 3 word val, 3 word test
      *  Computer Commands, 15 words train, 2 word val, 2 word test
      *  Common Words, 80 words train, 10 words val, 10 words test
      *  Uncommon Words, 240 words train, 30 words val, 30 words test
    This method ensures there are no overlapping code values between test and dev.
    @return 
    """
    test_codes = []
    dev_codes = []

    for code, val in codes.items():
        total_samples = int(math.ceil(val * 0.2))  # 20% of each category
        random_numbers = random.sample(range(1, val + 1), total_samples)  # Unique random numbers

        # Split into test/dev
        split_index = len(random_numbers) // 2
        test_numbers = random_numbers[:split_index]
        dev_numbers = random_numbers[split_index:]

        for num in test_numbers:
            if code == "UW":
                batch = random.randint(1, 3)  # Random batch selection for Uncommon Words
                test_codes.append(f"B{batch}_UW{num}")
            else:
                test_codes.append(f"{code}{num}")

        for num in dev_numbers:
            if code == "UW":
                batch = random.randint(1, 3)
                dev_codes.append(f"B{batch}_UW{num}")
            else:
                dev_codes.append(f"{code}{num}")

    assert not set(test_codes) & set(dev_codes), "Error: Test and Dev sets have overlapping elements!"

    return test_codes, dev_codes


def extract_code_from_id(rec_id: str) -> str:
    """
    Extracts the category and number from a given rec_id.
    Example:
      - "some_prefix_D2_suffix" -> "D2"
      - "random_L5_extra" -> "L5"
      - "B3_UW25_otherstuff" -> "B3_UW25"
    """
    match = re.search(r"(B[1-3]_UW\d+|D\d+|L\d+|C\d+|CW\d+|UW\d+)", rec_id)
    return match.group(0) if match else None


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
    Function to standardize the keys by removing the speaker code. Need this for finding
    intersection between typical and atypical speaker utterances. Typical speakers will start
    with a 'C'
    """
    if key[:1] =="C":
        return key[5:]
    return key[4:]


def load_speaker_utterances(metadata_mlf_path: str, speaker: str) -> Dict[str, str]:
    """
    Load utterances for a given speaker from an MLF file.
    """
    speaker_path = os.path.join(metadata_mlf_path, speaker, speaker + METADATA_FILE)
    assert os.path.isfile(speaker_path), f"No such file: {speaker_path}"
    return extract_mlf_information({}, speaker_path)


def find_intersecting_utterances(typical_utterances: Dict[str, str], atypical_utterances: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Find intersection of utterances between a typical speaker and an atypical speaker.
    Standardizes the keys before computing the intersection.
    """
    standardized_typical_keys = {standardize_key(k): k for k in typical_utterances}
    standardized_atypical_keys = {standardize_key(k): k for k in atypical_utterances}
    
    intersect_keys = set(standardized_typical_keys.keys()).intersection(set(standardized_atypical_keys.keys()))

    typical_filtered = {standardized_typical_keys[k]: typical_utterances[standardized_typical_keys[k]] for k in intersect_keys}
    atypical_filtered = {standardized_atypical_keys[k]: atypical_utterances[standardized_atypical_keys[k]] for k in intersect_keys}

    return typical_filtered, atypical_filtered


def process_utterances(
    atypical_speaker: str,
    typical_speaker: str,
    atypical_utterances: Dict[str, str],
    typical_utterances: Dict[str, str],
    corpus_audio_dir: Path,
    part: str
) -> Tuple[List[Recording], List[SupervisionSegment], List[Recording], List[SupervisionSegment]]:
    """
    Process matched utterances and create Recording and SupervisionSegment objects.
    """
    atypical_recordings, atypical_supervisions = [], []
    typical_recordings, typical_supervisions = [], []

    for (atypical_key, atypical_value), (typical_key, typical_value) in tqdm(zip(atypical_utterances.items(), typical_utterances.items()), desc="Processing Utterances"):
        try:
            atypical_recording_id = f"{atypical_value}_{atypical_key}"                       
            atypical_audio_path = corpus_audio_dir / part / atypical_speaker / f"{atypical_key}.wav"                            
            atypical_recording = Recording.from_file(atypical_audio_path, atypical_recording_id)

            typical_recording_id = f"{typical_value}_{typical_key}"
            typical_audio_path = corpus_audio_dir / part / typical_speaker / f"{typical_key}.wav"
            typical_recording = Recording.from_file(typical_audio_path, typical_recording_id)

            atypical_segment = SupervisionSegment(
                id=atypical_recording_id,
                recording_id=atypical_recording_id,
                start=0.0,
                duration=atypical_recording.duration,
                language="English",
                speaker=atypical_key,
                text=atypical_value
            )

            typical_segment = SupervisionSegment(
                id=typical_recording_id,
                recording_id=typical_recording_id,
                start=0.0,
                duration=typical_recording.duration,
                language="English",
                speaker=typical_key,
                text=typical_value
            )

            atypical_recordings.append(atypical_recording)
            atypical_supervisions.append(atypical_segment)
            typical_recordings.append(typical_recording)
            typical_supervisions.append(typical_segment)

        except Exception as err:
            logger.error(err)

    return atypical_recordings, atypical_supervisions, typical_recordings, typical_supervisions


def save_data(output_dir: Path, prefix: str, recordings: RecordingSet, supervisions: SupervisionSet):
    """
    Save recordings and supervisions to the output directory.
    """
    recordings.to_file(output_dir / f"uaspeech_recordings_{prefix}.jsonl.gz")
    supervisions.to_file(output_dir / f"uaspeech_supervisions_{prefix}.jsonl.gz")


def create_many_to_one_speaker_pair(
    corpus_dir: Pathlike,
    typical_speaker: str,
    atypical_speakers: List[str],
    alignments_dir: Optional[Pathlike] = None,
    dataset_parts: Union[str, Sequence[str]] = "auto",
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    corpus_dir = Path(corpus_dir)
    corpus_audio_dir = Path(os.path.join(corpus_dir, "audio"))
    alignments_dir = Path(alignments_dir) if alignments_dir is not None else corpus_audio_dir
    assert corpus_audio_dir.is_dir(), f"No such directory: {corpus_audio_dir}"

    if dataset_parts == "auto":
        dataset_parts = set(UASPEECH_FULL).intersection(path.name for path in corpus_audio_dir.glob("*"))
        if not dataset_parts:
            raise ValueError(f"Could not find any UASpeech dataset parts in: {corpus_audio_dir}")
    elif isinstance(dataset_parts, str):
        dataset_parts = [dataset_parts]

    logger.info(f"dataset_parts: {dataset_parts}")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    metadata_mlf_path = os.path.join(corpus_dir, "mlf")

    # Load the typical speaker's utterances
    typical_utterances = load_speaker_utterances(metadata_mlf_path, typical_speaker)

    # retrieve unique codes for the test and dev sets
    test_codes, dev_codes = generate_test_dev_utterances() 

    # Dynamically create storage for train, test, and dev sets
    splits = ["train", "test", "dev"]
    recording_sets = {f"atypical_recording_{split}_set": [] for split in splits}
    supervision_sets = {f"atypical_supervision_{split}_set": [] for split in splits}

    # Store typical recordings separately to prevent duplicates
    # Cannot have repeat recordings in a Recording Set
    typical_recording_sets = {split: {} for split in splits}
    typical_supervision_sets = {split: {} for split in splits}

    with ThreadPoolExecutor(num_jobs) as ex:
        for part in tqdm(dataset_parts, desc="Dataset parts"):
            if not part.startswith("."):
                logging.info(f"Processing UASpeech subset: {part}")

                for atypical_speaker in tqdm(atypical_speakers, desc="Processing Atypical Speakers"):
                    atypical_utterances = load_speaker_utterances(metadata_mlf_path, atypical_speaker)
                    typical_filtered, atypical_filtered = find_intersecting_utterances(typical_utterances, atypical_utterances)

                    assert len(typical_filtered) == len(atypical_filtered), (
                        f"Length Mismatch... Typical: {len(typical_filtered)} and Atypical: {len(atypical_filtered)}"
                    )

                    # Process the utterances
                    a_recs, a_sups, t_recs, t_sups = process_utterances(
                        atypical_speaker, typical_speaker, atypical_filtered, typical_filtered, corpus_audio_dir, part
                    )

                    # Determine split based on utterance ID
                    for a_rec, a_sup, t_rec, t_sup in zip(a_recs, a_sups, t_recs, t_sups):
                        extracted_id = extract_code_from_id(a_rec.id)

                        if extracted_id in test_codes:
                            split = "test"
                        elif extracted_id in dev_codes:
                            split = "dev"
                        else:
                            split = "train"

                        # Store in correct set
                        recording_sets[f"atypical_recording_{split}_set"].append(a_rec)
                        supervision_sets[f"atypical_supervision_{split}_set"].append(a_sup)
                        
                        # Ensure typical recordings are stored once per split
                        if t_rec.id not in typical_recording_sets[split]:
                            typical_recording_sets[split][t_rec.id] = t_rec
                            typical_supervision_sets[split][t_sup.id] = t_sup

    # Convert typical set to list
    for split in splits:
        recording_sets[f"typical_recording_{split}_set"] = list(typical_recording_sets[split].values())
        supervision_sets[f"typical_supervision_{split}_set"] = list(typical_supervision_sets[split].values())

    # Save data for all splits
    if output_dir is not None:
        for split in splits:
            save_data(output_dir, f"atypical_{split}", 
                      RecordingSet.from_recordings(recording_sets[f"atypical_recording_{split}_set"]),
                      SupervisionSet.from_segments(supervision_sets[f"atypical_supervision_{split}_set"]))
            
            save_data(output_dir, f"typical_{split}", 
                      RecordingSet.from_recordings(recording_sets[f"typical_recording_{split}_set"]),
                      SupervisionSet.from_segments(supervision_sets[f"typical_supervision_{split}_set"]))

    # Return dictionary of results
    return {
        f"atypical_{split}_recordings": RecordingSet.from_recordings(recording_sets[f"atypical_recording_{split}_set"])
        for split in splits
    } | {
        f"atypical_{split}_supervisions": SupervisionSet.from_segments(supervision_sets[f"atypical_supervision_{split}_set"])
        for split in splits
    } | {
        f"typical_{split}_recordings": RecordingSet.from_recordings(recording_sets[f"typical_recording_{split}_set"])
        for split in splits
    } | {
        f"typical_{split}_supervisions": SupervisionSet.from_segments(supervision_sets[f"typical_supervision_{split}_set"])
        for split in splits
    }


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

    typical_supervision_train_set = []
    atypical_supervision_train_set = []
    typical_supervision_test_set = []
    atypical_supervision_test_set = []

    with ThreadPoolExecutor(num_jobs) as ex:
        for part in tqdm(dataset_parts, desc="Dataset parts"):
            if not part.startswith("."):  # Avoid hidden files
                logging.info(f"Processing UASpeech subset: {part}")
                if manifests_exist(part=part, output_dir=output_dir):
                    logging.info(f"UASpeech subset: {part} already prepared - skipping.")
                    continue

                for typical_speaker, atypical_speaker in tqdm(zip(control_speakers, atypical_speakers), desc="Preparing Speaker Pair"):
                    # First retrieve path to individual speaker information for typical and atypical speakers
                    typical_speaker_path = os.path.join(metadata_mlf_path, typical_speaker, typical_speaker + METADATA_FILE)
                    atypical_speaker_path = os.path.join(metadata_mlf_path, atypical_speaker, atypical_speaker + METADATA_FILE)

                    assert os.path.isfile(typical_speaker_path), f"No such file: {typical_speaker_path}"
                    assert os.path.isfile(atypical_speaker_path), f"No such file: {atypical_speaker_path}"
                    # retrieves utterance code and word uttered. Inserts into a dictionary
                    typical_utterances = extract_mlf_information({}, typical_speaker_path)
                    atypical_utterances = extract_mlf_information({}, atypical_speaker_path)

                    # Removes speaker from key (i.e. CF02 and F02) to match utterance codes between atypical and typical speakers
                    standardize_typical_keys = {standardize_key(key): key for key in typical_utterances}
                    standardize_atypical_keys = {standardize_key(key): key for key in atypical_utterances}
                    # find intersection between dictionary items (ensures we have the same words uttered between speakers)
                    intersection_utterances = set(standardize_typical_keys.keys()).intersection(set(standardize_atypical_keys.keys()))

                    typical_utterances = {standardize_typical_keys[key]: typical_utterances[standardize_typical_keys[key]] for key in intersection_utterances}
                    atypical_utterances = {standardize_atypical_keys[key]: atypical_utterances[standardize_atypical_keys[key]] for key in intersection_utterances}
                    
                    # Check typical and atypical are the same length
                    assert len(typical_utterances) == len(atypical_utterances), f"Length Mismatch... Typical: {len(typical_utterances)} and Atypical: {len(atypical_utterances)}"
                    logger.info(f"Length of typical: {len(typical_utterances)}, Length of atypical: {len(atypical_utterances)}")

                    # Key: speaker code
                    # value: Utterance (Word)
                    for (atypical_key, atypical_value), (typical_key, typical_value) in tqdm(zip(atypical_utterances.items(), typical_utterances.items()), desc="Preparing parallel speakers"):
                        try:
                            atypical_recording_id = atypical_value + "_" + atypical_key                       
                            atypical_audio_path = corpus_audio_dir / part / atypical_speaker / f"{atypical_key}.wav"                            
                            atypical_recording = Recording.from_file(atypical_audio_path, atypical_recording_id)

                            typical_recording_id = typical_value + "_" + typical_key
                            typical_audio_path = corpus_audio_dir / part / typical_speaker / f"{typical_key}.wav"
                            typical_recording = Recording.from_file(typical_audio_path, typical_recording_id)
                            # Add supervision segment so vall-e can process text inputs
                            atypical_segment = SupervisionSegment(
                                id=atypical_recording_id,
                                recording_id=atypical_recording_id,
                                start=0.0,
                                duration=atypical_recording.duration,
                                language="English",
                                speaker=atypical_key,
                                text=atypical_value
                            )

                            typical_segment = SupervisionSegment(
                                id=typical_recording_id,
                                recording_id=typical_recording_id,
                                start=0.0,
                                duration=typical_recording.duration,
                                language="English",
                                speaker=typical_key,
                                text=typical_value
                            )
                            # test set is comprised of utterances in batch 2 else it's train set
                            if "_B2_" in atypical_recording_id and "_B2_" in typical_recording_id:
                                atypical_recording_test_set.append(atypical_recording)
                                atypical_supervision_test_set.append(atypical_segment)
                                
                                typical_recording_test_set.append(typical_recording)
                                typical_supervision_test_set.append(typical_segment)
                            else:
                                atypical_recording_train_set.append(atypical_recording)
                                atypical_supervision_train_set.append(atypical_segment)

                                typical_recording_train_set.append(typical_recording)
                                typical_supervision_train_set.append(typical_segment)

                        except Exception as err:
                            logger.error(err)

    atypical_recording_train_set = RecordingSet.from_recordings(atypical_recording_train_set)
    atypical_supervision_train_set = SupervisionSet.from_segments(atypical_supervision_train_set)
    
    typical_recording_train_set = RecordingSet.from_recordings(typical_recording_train_set)
    typical_supervision_train_set = SupervisionSet.from_segments(typical_supervision_train_set)

    atypical_recording_test_set = RecordingSet.from_recordings(atypical_recording_test_set)
    atypical_supervision_test_set = SupervisionSet.from_segments(atypical_supervision_test_set)
    
    typical_recording_test_set = RecordingSet.from_recordings(typical_recording_test_set)
    typical_supervision_test_set = SupervisionSet.from_segments(typical_supervision_test_set)

    if output_dir is not None:
        atypical_recording_train_set.to_file(output_dir / f"uaspeech_recordings_atypical_speakers_train.jsonl.gz")
        atypical_supervision_train_set.to_file(output_dir / f"uaspeech_supervisions_atypical_speakers_train.jsonl.gz")

        typical_recording_train_set.to_file(output_dir / f"uaspeech_recordings_typical_speakers_train.jsonl.gz")
        typical_supervision_train_set.to_file(output_dir / f"uaspeech_supervisions_typical_speakers_train.jsonl.gz")

        atypical_recording_test_set.to_file(output_dir / f"uaspeech_recordings_atypical_speakers_test.jsonl.gz")
        atypical_supervision_test_set.to_file(output_dir / f"uaspeech_supervisions_atypical_speakers_test.jsonl.gz")

        typical_recording_test_set.to_file(output_dir / f"uaspeech_recordings_typical_speakers_test.jsonl.gz")
        typical_supervision_test_set.to_file(output_dir / f"uaspeech_supervisions_typical_speakers_test.jsonl.gz")

    return {
        "atypical_train_recordings": atypical_recording_train_set,
        "atypical_train_supervisions": atypical_supervision_train_set,
        "typical_train_recordings": typical_recording_train_set,
        "typical_train_supervisions": typical_supervision_train_set,   
        "atypical_test_recordings": atypical_recording_test_set,
        "atypical_test_supervisions": typical_supervision_test_set,
        "typical_test_recordings": typical_recording_test_set,
        "typical_test_supervisions": typical_supervision_test_set,
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
# control_speakers = ["CF02", "CF03", "CF04", "CM04", "CM05", "CM06", "CM08", "CM10", "CM12", "CM13"]
# atypical_speakers = ["F02", "F03", "F04", "M04", "M05", "M07", "M08", "M10", "M11", "M12"]

control_speakers = ["CF02", "CF04", "CM12", "CM06", "CM10"]
atypical_speakers = ["CF02", "CF04", "CM12", "CM06", "CM10"]

# create_speaker_speaker_pair(UASPEECH_PATH, control_speakers, atypical_speakers, None, "normalized", output_dir="/home/data1/vall-e.git/VallE/egs/uaspeech/data/manifests")

create_many_to_one_speaker_pair(UASPEECH_PATH, "CM05", atypical_speakers, None, "normalized", output_dir="/home/data1/vall-e.git/VallE/egs/uaspeech/data/manifests")

    
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