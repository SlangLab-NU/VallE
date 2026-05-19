import json, gzip, argparse, os, logging
from pathlib import Path

log_path = os.getcwd() + "/inference_text.log"
logging.basicConfig(
    filename=log_path,
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--testset-path",
        type=Path,
        default=Path("data/tokenized/cuts_test.jsonl.gz"),
        help="Path to the tokenized test CutSet (cuts_test.jsonl.gz)",
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("sap_inference_list.txt"),
        help="Output path for the generated inference txt file",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="",
        help="Experiment directory. Synthesized wavs will be saved under {exp_dir}/infer/",
    )

    # Optional filtering args
    parser.add_argument(
        "--ratings-csv",
        type=Path,
        default=None,
        help=(
            "Path to speaker ratings CSV (columns: Speaker_ID, Etiology, Average_Rating, Split, …). "
            "Required when using --etiology, --min-rating, or --max-rating."
        ),
    )

    parser.add_argument(
        "--etiology",
        type=str,
        default=None,
        help="Only include speakers with this etiology (e.g. \"Parkinson's Disease\").",
    )

    parser.add_argument(
        "--min-rating",
        type=float,
        default=None,
        help="Only include speakers whose Average_Rating >= this value.",
    )

    parser.add_argument(
        "--max-rating",
        type=float,
        default=None,
        help="Only include speakers whose Average_Rating <= this value.",
    )

    parser.add_argument(
        "--speakers",
        type=str,
        default=None,
        help="Comma-separated list of speaker UUIDs to include. Overrides CSV-based filters.",
    )

    return parser.parse_args()


def load_speaker_filter(args):
    """
    Returns a set of allowed speaker IDs, or None if no filtering is requested.
    Explicit --speakers takes priority; otherwise the ratings CSV is used.
    """
    if args.speakers:
        allowed = set(s.strip() for s in args.speakers.split(",") if s.strip())
        logger.info(f"Filtering to {len(allowed)} explicitly specified speakers.")
        return allowed

    if args.ratings_csv is None:
        return None  # no filter — include all speakers

    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas is required for CSV-based filtering. Install it or use --speakers instead.")
        raise

    df = pd.read_csv(args.ratings_csv)

    if args.etiology:
        df = df[df["Etiology"] == args.etiology]
        logger.info(f"Filtered to etiology '{args.etiology}': {len(df)} speakers.")

    if args.min_rating is not None:
        df = df[df["Average_Rating"] >= args.min_rating]
        logger.info(f"Filtered to Average_Rating >= {args.min_rating}: {len(df)} speakers.")

    if args.max_rating is not None:
        df = df[df["Average_Rating"] <= args.max_rating]
        logger.info(f"Filtered to Average_Rating <= {args.max_rating}: {len(df)} speakers.")

    allowed = set(df["Speaker_ID"].tolist())
    logger.info(f"Final speaker filter: {len(allowed)} speakers.")
    return allowed


def create_inference_txt(jsonl_gz_path, output_path, exp_dir, allowed_speakers):
    infer_dir = os.path.join(exp_dir, "infer") if exp_dir else "infer"
    os.makedirs(infer_dir, exist_ok=True)

    lines = []
    seen_per_speaker = {}

    with gzip.open(str(jsonl_gz_path), "rt") as f:
        for raw in f:
            recording = json.loads(raw)
            supervisions = recording["supervisions"][0]
            speaker = supervisions["speaker"]

            if allowed_speakers is not None and speaker not in allowed_speakers:
                continue

            transcript = supervisions["text"]

            # De-duplicate: skip if we've already seen this exact transcript for this speaker
            if speaker not in seen_per_speaker:
                seen_per_speaker[speaker] = set()
            if transcript in seen_per_speaker[speaker]:
                continue
            seen_per_speaker[speaker].add(transcript)

            utt_id = supervisions["id"]
            source_audio = recording["recording"]["sources"][0]["source"]
            output_wav = os.path.join(os.getcwd(), infer_dir, f"{utt_id}_synthesized.wav")

            lines.append(f"{transcript}\t{source_audio}\t{transcript}\t{output_wav}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    total_speakers = len(seen_per_speaker)
    logger.info(f"Wrote {len(lines)} inference entries across {total_speakers} speakers → {output_path}")
    logger.info(f"Synthesized wavs will go to: {infer_dir}")

    for spk, utts in sorted(seen_per_speaker.items()):
        logger.info(f"  {spk}: {len(utts)} utterances")

    print(f"Wrote {len(lines)} lines for {total_speakers} speakers → {output_path}")


def main():
    args = get_parser()

    allowed_speakers = load_speaker_filter(args)
    create_inference_txt(
        args.testset_path,
        args.output_path,
        args.exp_dir,
        allowed_speakers,
    )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
