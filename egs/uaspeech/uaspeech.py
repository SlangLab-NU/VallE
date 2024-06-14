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
    typical_speaker: str,
    atypical_speaker: str,
    alignments_dir: Optional[Pathlike] = None,
    dataset_parts: Union[str, Sequence[str]] = "auto",
    output_dir: Optional[Pathlike] = None, 
    num_jobs: int = 1,
)-> Dict[str, Dict[str, Union[RecordingSet, RecordingSet]]]:
    """
    Returns a manifest which consists of a RecordingSet pair. The pair contains a typical speaker
    and an atypical speaker linking to an audio recording of the same utterance. This is used for establishing 
    parallel model training.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param dataset_parts: string or sequence of strings representing dataset part names, e.g. 'normalized', 'noisereduced', 'original'.
        By default we will infer which parts are available in ``corpus_dir``.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param num_jobs: int, number of parallel threads used for 'parse_utterance' calls.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'typical_audio' and 'atypical_audio'.
    """
    corpus_dir = Path(corpus_dir)
    corpus_audio_dir = Path(os.path.join(corpus_dir, "audio"))
    alignments_dir = Path(alignments_dir) if alignments_dir is not None else corpus_audio_dir
    assert corpus_audio_dir.is_dir(), f"No such directory: {corpus_audio_dir}"

    if dataset_parts == "auto":
        print("I went down the auto path")
        dataset_parts = set(UASPEECH_FULL).intersection(path.name for path in corpus_audio_dir.glob("*"))
        
        if not dataset_parts:
            raise ValueError(
                f"Could not find any of UASpeech dataset parts in: {corpus_audio_dir}"
            )
    # Here we can specify a specific processed batch of speaker data
    elif isinstance(dataset_parts, str):
        print("dataset = " + dataset_parts)
        dataset_parts = [dataset_parts]
    logger.info(f"dataset_parts: {dataset_parts}")
    manifests = {}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Maybe the manifests already exist: we can read them and save a bit of preparation time.
        manifests = read_manifests_if_cached(
            dataset_parts=dataset_parts, output_dir=output_dir
        )

    metadata_mlf_path = os.path.join(corpus_dir, "mlf")

    typical_recording_train_set = []
    atypical_recording_train_set = []
    typical_recording_test_set = []
    atypical_recording_test_set = []

    with ThreadPoolExecutor(num_jobs) as ex:
        for part in tqdm(dataset_parts, desc="Dataset parts"):
            if not part.startswith("."): # Avoid hidden files
                logging.info(f"Processing UASpeech subset: {part}")
                # Check if part has been prepared already
                if manifests_exist(part=part, output_dir=output_dir):
                    logging.info(f"UASpeech subset: {part} already prepared - skipping.")
                    continue
                
                # assign metadata file for speaker
                typical_speaker_path =  os.path.join(os.path.join(metadata_mlf_path, typical_speaker), typical_speaker + METADATA_FILE)
                atypical_speaker_path = os.path.join(os.path.join(metadata_mlf_path, atypical_speaker), atypical_speaker + METADATA_FILE)
                
                # Ensure both metadata files exist
                assert os.path.isfile(typical_speaker_path), f"No such file: {typical_speaker_path}"
                assert os.path.isfile(atypical_speaker_path), f"No such file: {atypical_speaker_path}"

                typical_utterances = {}
                atypical_utterances = {}

                typical_utterances = extract_mlf_information(typical_utterances, typical_speaker_path)
                atypical_utterances = extract_mlf_information(atypical_utterances, atypical_speaker_path)

                # Remove 'C' from typical utterances
                standardize_typical_keys = {standardize_key(key): key for key in typical_utterances}
                # Find intersection between two dictionaries
                intersection_utterances = set(standardize_typical_keys.keys()).intersection(set(atypical_utterances.keys()))
                # Set typical utterances back to including 'C'
                typical_utterances = {standardize_typical_keys[key]: typical_utterances[standardize_typical_keys[key]] for key in intersection_utterances}
                atypical_utterances = {key: atypical_utterances[key] for key in intersection_utterances}
                
                assert len(typical_utterances) == len(atypical_utterances), f"Length Mismatch... Typical: {len(typical_utterances)} and Atypical: {len(atypical_utterances)}"
                logger.info(f"Length of typical: {len(typical_utterances)}, Length of atypical: {len(atypical_utterances)}")

                for key, value in tqdm(atypical_utterances.items(), desc="Preparing parallel speakers"):
                    # Since the two dictionaries are the same other than a 'C' in front of the key for typical speaker
                    # This iterates through one dictionary to create recording set for both speakers
                    try:
                        atypical_recording_id = value + "_" + key
                        atypical_audio_path = corpus_audio_dir / part / atypical_speaker /f"{key}.wav"
                        atypical_recording = Recording.from_file(
                            atypical_audio_path, 
                            atypical_recording_id
                        )
                        if "_B2_" in atypical_recording_id:
                            atypical_recording_test_set.append(atypical_recording)
                        else:
                            atypical_recording_train_set.append(atypical_recording)

                        typical_recording_id = value + "_C" + key
                        typical_audio_path = corpus_audio_dir / part / typical_speaker /f"C{key}.wav"
                        typical_recording = Recording.from_file(
                            typical_audio_path,
                            typical_recording_id
                        )
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

    # Cannot utilize fix_manifests as it requires a supervision. I could add one to ensure recording is validated and remove it later
    # typical_recording_set= fix_manifests(typical_recording_set)

    # Need to separate control recordings and impaired speakers
    if output_dir is not None:
        typical_recording_train_set.to_file(output_dir / "CF02_train_recordings.jsonl.gz")
        atypical_recording_train_set.to_file(output_dir / "F02_train_recordings.jsonl.gz")
        typical_recording_test_set.to_file(output_dir / "CF02_test_recordings.jsonl.gz")
        atypical_recording_test_set.to_file(output_dir / "F02_test_recordings.jsonl.gz")

    return {"typical_train_recordings": typical_recording_train_set,
            "atypical_train_recordings": atypical_recording_train_set,
            "typical_test_recordings": typical_recording_test_set,
            "atypical_test_recordings": atypical_recording_test_set}

def prepare_uaspeech(
    corpus_dir: Pathlike,
    alignments_dir: Optional[Pathlike] = None,
    dataset_parts: Union[str, Sequence[str]] = "auto",
    output_dir: Optional[Pathlike] = None,
    # normalize_text: str = "none",
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param alignments_dir: Pathlike, the path of the alignments dir. By default, it is
        the same as ``corpus_dir``.
    :param dataset_parts: string or sequence of strings representing dataset part names, e.g. 'normalized', 'noisereduced', 'original'.
        By default we will infer which parts are available in ``corpus_dir``.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param normalize_text: str, "none" or "lower",
        for "lower" the transcripts are converted to lower-case.
    :param num_jobs: int, number of parallel threads used for 'parse_utterance' calls.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    # In our case the corpus_dir may have to be hardcoded as a global VAR 
    # Have to split dir into audio branch and metadata
    corpus_dir = Path(corpus_dir)
    corpus_audio_dir = Path(os.path.join(corpus_dir, "audio"))
    alignments_dir = Path(alignments_dir) if alignments_dir is not None else corpus_audio_dir
    assert corpus_audio_dir.is_dir(), f"No such directory: {corpus_audio_dir}"

    if dataset_parts == "auto":
        print("I went down the auto path")
        dataset_parts = set(UASPEECH_FULL).intersection(path.name for path in corpus_audio_dir.glob("*"))
        
        if not dataset_parts:
            raise ValueError(
                f"Could not find any of UASpeech dataset parts in: {corpus_audio_dir}"
            )
    # Here we can specify a specific processed batch of speaker data
    elif isinstance(dataset_parts, str):
        print("dataset = " + dataset_parts)
        dataset_parts = [dataset_parts]
    print(dataset_parts)
    manifests = {}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Maybe the manifests already exist: we can read them and save a bit of preparation time.
        manifests = read_manifests_if_cached(
            dataset_parts=dataset_parts, output_dir=output_dir
        )

    # The file layers of UaSpeech are as follows:
    # ["noisereduced", "normalized", "original"]
    #    |
    #     ---> Controls: CF02 CF03 CF04 CF05 CM01 CM04 CM05 CM06 CM08 CM09 CM10 CM12 CM13
    #     ---> Cerebral Speakers: F02 F03 F04 F05 M01 M04 M05 M07 M08 M09 M10 M11 M12 M14 M16 (28 speakers total)
    #       |
    #        ---> wav files for each speaker (Ex CF02 contains 5355 items, some speakers contain less)

    metadata_mlf_path = os.path.join(corpus_dir, "mlf")
    # assert metadata_csv_path.is_file(), f"No such file: {metadata_csv_path}"

    # Generate a mapping (Where the audio data, text will be)

    train_control_recordings = []
    train_control_supervisions = []

    test_control_recordings = []
    test_control_supervisions = []
    
    train_cerebral_recordings = []
    train_cerebral_supervisions = []

    test_cerebral_recordings = []
    test_cerebral_supervisions = []

    # Updated version
    with ThreadPoolExecutor(num_jobs) as ex:
        for part in tqdm(dataset_parts, desc="Dataset parts"):
            if not part.startswith("."): # Avoid hidden files
                logging.info(f"Processing UASpeech subset: {part}")
                # Check if part has been prepared already
                if manifests_exist(part=part, output_dir=output_dir):
                    logging.info(f"UASpeech subset: {part} already prepared - skipping.")
                    continue
                for speaker in tqdm(os.listdir(metadata_mlf_path), desc=f"Preparing speaker {part}\n"):
                    print(f"SPEAKER: {speaker}")
                    metadata_utterances = os.path.join(os.path.join(metadata_mlf_path, speaker), speaker + METADATA_FILE)
                    # assert metadata_utterances.is_file(), f"No such file: {metadata_utterances}"
                    with open(metadata_utterances, 'r') as file:
                        lines = file.readlines()
                        current_entry = None
                        text= None
                        i = 0
                        for line in lines:
                            line = line.strip()
                            # Skip comments or empty lines
                            if line.startswith("#") or line.startswith(".") or not line:
                                continue
                            # Start of a new entry
                            if line.startswith('"'):
                                # line[3:-5] removes the '*/' and '.lab'
                                current_entry = line[3:-5]
                                audio_path = corpus_audio_dir / part / speaker / f"{current_entry}.wav"
                                if not audio_path.is_file():
                                    logging.warning(f"No such file: {audio_path}")
                                    continue
                            else:
                                # Parse sequence information
                                text=line
                            try:
                                recording_id = str(i) + "_"+ part + "_" + current_entry                               
                                recording = Recording.from_file(
                                    audio_path,
                                    recording_id=recording_id
                                )                               
                                segment = SupervisionSegment(
                                    id=recording_id,
                                    recording_id=recording_id,
                                    start=0.0,
                                    duration=recording.duration,
                                    language="English",
                                    gender=speaker,
                                    text=text
                                )
                                if text != None:
                                    if current_entry.startswith("C"):
                                        if "_B2_" in recording_id:
                                            test_control_recordings.append(recording)
                                            test_control_supervisions.append(segment)
                                        else:
                                            train_control_recordings.append(recording)
                                            train_control_supervisions.append(segment)
                                        # print(recording)
                                        # print(segment)
                                    else:
                                        # Check if it's a test recording
                                        if "_B2_" in recording_id:
                                            test_cerebral_recordings.append(recording)
                                            test_cerebral_supervisions.append(segment)
                                        else:
                                            if text != None:
                                                train_cerebral_recordings.append(recording)
                                                train_cerebral_supervisions.append(segment)
                                        # print(recording)
                                        # print(segment)
                                else:
                                    print(recording)
                                    print(segment)
                            except Exception as err:
                                logger.error(err)
                            i += 1
        
        # Create train and test sets for control and cerebral recordings
        train_control_recording_set = RecordingSet.from_recordings(train_control_recordings)
        train_control_supervision_set = SupervisionSet.from_segments(train_control_supervisions)

        test_control_recording_set = RecordingSet.from_recordings(test_control_recordings)
        test_control_supervision_set = SupervisionSet.from_segments(test_control_supervisions)

        train_cerebral_recording_set = RecordingSet.from_recordings(train_cerebral_recordings)
        train_cerebral_supervision_set = SupervisionSet.from_segments(train_cerebral_supervisions)

        test_cerebral_recording_set = RecordingSet.from_recordings(test_cerebral_recordings)
        test_cerebral_supervision_set = SupervisionSet.from_segments(test_cerebral_supervisions)

        """
        Sample outputs:

        Recording(id='4947_normalized_M11_B3_LR_M7', sources=[AudioSource(type='file', channels=[0], source='/home/data1/data/UASpeech/audio/normalized/M11/M11_B3_LR_M7.wav')], 
                    sampling_rate=16000, num_samples=50675, duration=3.1671875, channel_ids=[0], transforms=None)
        
        SupervisionSegment(id='4947_normalized_M11_B3_LR_M7', recording_id='4947_normalized_M11_B3_LR_M7', start=0.0, duration=3.1671875, channel=0, text='ROMEO', language='English', 
                            speaker=None, gender='M11', custom=None, alignment=None)
        
        Recording(id='4948_normalized_M11_B1_CW11_M7', sources=[AudioSource(type='file', channels=[0], source='/home/data1/data/UASpeech/audio/normalized/M11/M11_B1_CW11_M7.wav')], 
                    sampling_rate=16000, num_samples=55457, duration=3.4660625, channel_ids=[0], transforms=None)
        
        SupervisionSegment(id='4948_normalized_M11_B1_CW11_M7', recording_id='4948_normalized_M11_B1_CW11_M7', start=0.0, duration=3.4660625, channel=0, text='ROMEO', language='English', 
                            speaker=None, gender='M11', custom=None, alignment=None)
        """

        train_control_recording_set, train_control_supervision_set = fix_manifests(train_control_recording_set, train_control_supervision_set)
        validate_recordings_and_supervisions(train_control_recording_set, train_control_supervision_set)

        test_control_recording_set, test_control_supervision_set = fix_manifests(test_control_recording_set, test_control_supervision_set)
        validate_recordings_and_supervisions(test_control_recording_set, test_control_supervision_set)

        train_cerebral_recording_set, train_cerebral_supervision_set = fix_manifests(train_cerebral_recording_set, train_cerebral_supervision_set)
        validate_recordings_and_supervisions(train_cerebral_recording_set, train_cerebral_supervision_set)

        test_cerebral_recording_set, test_cerebral_supervision_set = fix_manifests(test_cerebral_recording_set, test_cerebral_supervision_set)
        validate_recordings_and_supervisions(test_cerebral_recording_set, test_cerebral_supervision_set)

        # Need to separate control recordings and impaired speakers
        if output_dir is not None:
            train_control_supervision_set.to_file(output_dir / "uaspeech_supervisions_train_control.jsonl.gz")
            train_control_recording_set.to_file(output_dir / "uaspeech_recordings_train_control.jsonl.gz")

            test_control_supervision_set.to_file(output_dir / "uaspeech_supervisions_test_control.jsonl.gz")
            test_control_recording_set.to_file(output_dir / "uaspeech_recordings_test_control.jsonl.gz")

            train_cerebral_supervision_set.to_file(output_dir / "uaspeech_supervisions_train_cerebral.jsonl.gz")
            train_cerebral_recording_set.to_file(output_dir / "uaspeech_recordings_train_cerebral.jsonl.gz")

            test_cerebral_supervision_set.to_file(output_dir / "uaspeech_supervisions_test_cerebral.jsonl.gz")
            test_cerebral_recording_set.to_file(output_dir / "uaspeech_recordings_test_cerebral.jsonl.gz")


        return {"train_recordings_control": train_control_recording_set, "train_supervisions_control": train_control_supervision_set,
                "test_recordings_control": test_control_recording_set, "test_supervisions_control": test_control_supervision_set,
                "train_recordings_cerebral": train_cerebral_recording_set, "train_supervisions_cerebral": train_cerebral_supervision_set,
                "test_recordings_cerebral": test_cerebral_recording_set, "test_supervisions_cerebral": test_cerebral_supervision_set}


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

sets = create_speaker_speaker_pair(UASPEECH_PATH, "CF02", "F02", None, "normalized", output_dir="/home/data1/vall-e.git/VallE/egs/uaspeech/data/manifests")
typical_train = sets["typical_train_recordings"]
atypical_train = sets["atypical_train_recordings"]
typical_test = sets["typical_test_recordings"]
atypical_test = sets["atypical_test_recordings"]

print(f"Typical train length {len(typical_train)}")
print(f"Atypical train length {len(atypical_train)}")
print(f"Typical test length {len(typical_test)}")
print(f"Atypical test length {len(atypical_test)}")


# sets = prepare_uaspeech(UASPEECH_PATH, None, "normalized", output_dir="/home/data1/vall-e.git/VallE/egs/uaspeech/data/manifests")
# train_cerebral = sets["train_supervisions_cerebral"]

# for element in train_cerebral:
#     print(element)
############################################################################################