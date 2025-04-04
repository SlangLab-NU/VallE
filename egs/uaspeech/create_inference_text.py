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
        "--atyp-to-atyp-output-path",
        type=Path,
        default=Path('atyp_to_atyp_inference_list.txt'),
        help="Output path for the generated txt file",
    )

    parser.add_argument(
        "--atyp-to-typ-output-path",
        type=Path,
        default=Path('atyp_to_typ_inference_list.txt'),
        help="Output path for the generated txt file",
    )

    parser.add_argument(
        "--atyp-speakers",
        type=str,
        default="very_low",
        help="Comma-separated list of speakers to perform inference on",
    )

    parser.add_argument(
        "--typical-inference",
        type=int,
        choices=[0,1],
        default=0,
        help="0 if we are running inference on typical speakers 1 if typical",
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

def create_inference_txt_dual(
    jsonl_gz_path, atyp_speakers, typ_speakers_map, exp_dir,
    atyp_to_atyp_output_path, atyp_to_typ_output_path
):
    infer_dir = os.path.join(exp_dir, "infer")
    os.makedirs(infer_dir, exist_ok=True)

    atyp_to_atyp_lines = []
    atyp_to_typ_lines = []

    seen_words_per_speaker = {spk: set() for spk in atyp_speakers}

    with gzip.open(str(jsonl_gz_path), 'rt') as f:
        print("opened file")
        for line in f:
            recording = json.loads(line)
            supervisions = recording['supervisions'][0]
            speaker = supervisions['speaker'].split('_')[0]
            if speaker not in atyp_speakers:
                continue    

            prompt_text = supervisions['text']
            if prompt_text in seen_words_per_speaker[speaker]:
                continue
            seen_words_per_speaker[speaker].add(prompt_text)

            utt_id = supervisions['id']
            prompt_audio = recording['recording']['sources'][0]['source']
            text_to_synthesize = prompt_text

            # ------------------------
            # 1) atyp-to-atyp entry
            # ------------------------
            output_path_atyp = os.path.join(os.getcwd(), exp_dir, "infer", f"{utt_id}_synthesized.wav")
            line_atyp = f"{prompt_text}\t{prompt_audio}\t{text_to_synthesize}\t{output_path_atyp}"
            atyp_to_atyp_lines.append(line_atyp)

            # ------------------------
            # 2) atyp-to-typ entry
            # ------------------------
            typ_speaker = typ_speakers_map[speaker]
            typ_audio = prompt_audio.replace(f"/{speaker}/", f"/{typ_speaker}/").replace(f"{speaker}_", f"{typ_speaker}_")
            line_typ = f"{prompt_text}\t{output_path_atyp}\t{text_to_synthesize}\t{typ_audio}"
            atyp_to_typ_lines.append(line_typ)

    # Save both txt files
    with open(atyp_to_atyp_output_path, "w") as f1:
        f1.write("\n".join(atyp_to_atyp_lines))

    with open(atyp_to_typ_output_path, "w") as f2:
        f2.write("\n".join(atyp_to_typ_lines))

    logger.info(f"Generated {len(atyp_to_atyp_lines)} lines in {atyp_to_atyp_output_path}")
    logger.info(f"Generated {len(atyp_to_typ_lines)} lines in {atyp_to_typ_output_path}")
    logger.info(f"Inference outputs will go to: {infer_dir}")

def create_typical_inference_txt(jsonl_gz_path, typical_speakers, exp_dir, output_path):
    infer_dir = os.path.join(exp_dir, "infer")
    os.makedirs(infer_dir, exist_ok=True)

    output_lines = []

    seen_words_per_speaker = {spk: set() for spk in typical_speakers}
    with gzip.open(str(jsonl_gz_path), 'rt') as f:
        for line in f:
            recording = json.loads(line)
            supervisions = recording['supervisions'][0]
            speaker = supervisions['speaker'].split('_')[0]
            if speaker not in typical_speakers:
                continue

            prompt_text = supervisions['text'].lower()
            if prompt_text in seen_words_per_speaker[speaker]:
                continue
            seen_words_per_speaker[speaker].add(prompt_text)

            utt_id = supervisions['id']
            prompt_audio = recording['recording']['sources'][0]['source']
            text_to_synthesize = prompt_text
            synth_path = os.path.join(os.getcwd(), exp_dir, "infer", f"{utt_id}_synthesized.wav")

            line = f"{prompt_text}\t{prompt_audio}\t{text_to_synthesize}\t{synth_path}"
            output_lines.append(line)

    with open(output_path, "w") as f:
        print(output_path)
        f.write("\n".join(output_lines))

    logger.info(f"Generated {len(output_lines)} lines in {output_path}")
    logger.info(f"Inference outputs will go to: {infer_dir}")


def main():
    args = get_parser()

    if args.typical_inference == 0:
        very_low = ["F03", "M01","M04","M16"]
        low = ["F02","M07"]
        medium = ["F04","M05","M11"]

        atyp_speakers = []
        typ_speakers = []
        
        if args.atyp_speakers == 'very_low':
            atyp_speakers = very_low
            typ_speakers = ["CF03", "CM06", "CM04", "CM13"]
        elif args.atyp_speakers == 'low':
            atyp_speakers = low
            typ_speakers = ["CF02","CM08"]
        elif args.atyp_speakers == 'medium':
            atyp_speakers = medium
            typ_speakers = ["CF04","CM05","CM12"]

        # build the mapping
        typ_speakers_map = dict(zip(atyp_speakers, typ_speakers))

        create_inference_txt_dual(
            args.testset_path,
            atyp_speakers,
            typ_speakers_map,
            args.exp_dir,
            args.atyp_to_atyp_output_path,
            args.atyp_to_typ_output_path
        )
    
    else:
        print("Typical Inference")
        typical_speakers = ["CM08", "CM13", "CF02", "CF04"]  # or your test set

        create_typical_inference_txt(
            args.testset_path,
            typical_speakers,
            args.exp_dir,
            args.atyp_to_atyp_output_path,  # reuse this as the output path
        )

if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()