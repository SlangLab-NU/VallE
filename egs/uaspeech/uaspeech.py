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
                for speaker in tqdm(os.listdir(metadata_mlf_path), desc=f"Preparing speaker {part}"):
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
                                if current_entry.startswith("C"):
                                    if "_B2_" in recording_id:
                                        test_control_recordings.append(recording)
                                        test_control_supervisions.append(segment)
                                    else:
                                        train_control_recordings.append(recording)
                                        train_control_supervisions.append(segment)
                                else:
                                    # Check if it's a test recording
                                    if "_B2_" in recording_id:
                                        test_cerebral_recordings.append(recording)
                                        test_cerebral_supervisions.append(segment)
                                    else:
                                        train_cerebral_recordings.append(recording)
                                        train_cerebral_supervisions.append(segment)
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

        Recording(id='CF03_B3_UW93_M3', sources=[AudioSource(type='file', channels=[0], source='/home/data1/data/UASpeech/audio/normalized/CF03/CF03_B3_UW93_M3.wav')], 
                sampling_rate=16000, num_samples=33724, duration=2.10775, channel_ids=[0], transforms=None)
        
        SupervisionSegment(id='7648normalized_CF03_B3_UW93_M3', recording_id='7648normalized_CF03_B3_UW93_M3', start=0.0, duration=2.10775, 
                            channel=0, text='DISPOSSESS', language='English', speaker=None, gender='CF03', custom=None, alignment=None)
        
        Recording(id='CF03_B3_UW93_M3', sources=[AudioSource(type='file', channels=[0], source='/home/data1/data/UASpeech/audio/normalized/CF03/CF03_B3_UW93_M3.wav')], 
                 sampling_rate=16000, num_samples=33724, duration=2.10775, channel_ids=[0], transforms=None)
        
        SupervisionSegment(id='7649normalized_CF03_B3_UW93_M3', recording_id='7649normalized_CF03_B3_UW93_M3', start=0.0, duration=2.10775, 
                            channel=0, text='GLASSES', language='English', speaker=None, gender='CF03', custom=None, alignment=None)
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
            train_control_supervision_set.to_file(output_dir / "uaspeech_train_control_supervisions.jsonl.gz")
            train_control_recording_set.to_file(output_dir / "uaspeech_train_control_recordings.jsonl.gz")

            test_control_supervision_set.to_file(output_dir / "uaspeech_test_control_supervisions.jsonl.gz")
            test_control_recording_set.to_file(output_dir / "uaspeech_test_control_recordings.jsonl.gz")

            train_cerebral_supervision_set.to_file(output_dir / "uaspeech_train_cerebral_supervisions.jsonl.gz")
            train_cerebral_recording_set.to_file(output_dir / "uaspeech_train_cerebral_recordings.jsonl.gz")

            test_cerebral_supervision_set.to_file(output_dir / "uaspeech_test_cerebral_supervisions.jsonl.gz")
            test_cerebral_recording_set.to_file(output_dir / "uaspeech_test_cerebral_recordings.jsonl.gz")


        return {"train_control_recordings": train_control_recording_set, "train_control_supervisions": train_control_supervision_set,
                "test_control_recordings": test_control_recording_set, "test_control_supervisions": test_control_supervision_set,
                "train_cerebral_recordings": train_cerebral_recording_set, "train_cerebral_supervisions": train_cerebral_supervision_set,
                "test_cerebral_recordings": test_cerebral_recording_set, "test_cerebral_supervisions": test_cerebral_supervision_set}


# TEMPORARY TEST SCRIPT
###########################################################################################
# mlf_file_path = "/home/data1/data/UASpeech/mlf/CF04/CF04_word.mlf"

# mlf_data = read_mlf(mlf_file_path)
# # label, sequence = mlf_data
# # Print the result
# for entry in mlf_data:
#     print(entry)
    # print(f"Label: {entry['label']}")
    # print("Sequence:", end=" ")
    # for label in entry['sequence']:
    #     print(f"  {label}")
sets = prepare_uaspeech(UASPEECH_PATH, None, "normalized", output_dir="/home/data1/VallE/vall-e/egs/uaspeech/data/manifests")
print(sets)
############################################################################################