import json, gzip, argparse, os, logging
from pathlib import Path

log_path = os.getcwd() + "/inference_text.log"
logging.basicConfig(filename=log_path, format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug("This is a debug log")
logger.info("This is an info log")
logger.critical("This is critical")
logger.error("An error occurred")

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--testset-path",
        type=Path,
        default=Path('data/tokenized/cuts_test.jsonl.gz'),
        help="the path to the Test CutSet",
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path('inference_list.txt'),
        help="Output path for the generated txt file",
    )

    parser.add_argument(
        "--speakers",
        type=str,
        default="M12,M16,F02,F04",
        help="Comma-separated list of speakers to perform inference on",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    return parser.parse_args()

def create_inference_txt(jsonl_gz_path, speakers, exp_dir, output_txt_path):
    """
    Create a text file for infer.py batch mode and make sure infer directory exists.

    Args:
        jsonl_gz_path (str): Path to the cuts_test.jsonl.gz file
        speakers (list): List of speaker IDs to include (e.g., ['M04', 'F02'])
        exp_dir (str): The experiment directory (used as model name)
        output_txt_path (str): Output path for the txt file
    """

    infer_dir = os.path.join(exp_dir, "infer")
    os.makedirs(infer_dir, exist_ok=True)  # Automatically create if missing

    output_lines = []
    seen_words_per_speaker = {spk: set() for spk in speakers}
    with gzip.open(str(jsonl_gz_path), 'rt') as f:
        print("opened file")
        for line in f:
            recording = json.loads(line)
            supervisions = recording['supervisions'][0]
            speaker = supervisions['speaker'].split('_')[0]
            if speaker not in speakers:
                continue    
            prompt_text = supervisions['text']

            if prompt_text in seen_words_per_speaker[speaker]:
                continue
            seen_words_per_speaker[speaker].add(prompt_text)

            utt_id = supervisions['id']
            prompt_audio = recording['recording']['sources'][0]['source']
            text_to_synthesize = prompt_text
            output_path = os.path.join(exp_dir, "infer", f"{utt_id}_synthesized.wav")
            line = f"{prompt_text}\t{prompt_audio}\t{text_to_synthesize}\t{output_path}"
            output_lines.append(line)

    # Save txt
    with open(output_txt_path, "w") as out_f:
        out_f.write("\n".join(output_lines))

    logger.info(f"Generated {len(output_lines)} lines in {output_txt_path}")
    logger.info(f"Inference outputs will go to: {infer_dir}")

def main():
    args = get_parser()
    speakers = [s.strip() for s in args.speakers.split(",")]
    create_inference_txt(args.testset_path, speakers, args.exp_dir, args.output_path)

if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()