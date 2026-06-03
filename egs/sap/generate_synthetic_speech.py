import pandas as pd
import json
import logging
import sys
import os
from contextlib import contextmanager
from pathlib import Path
import argparse
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    logger.info("Downloading NLTK punkt_tab...")
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)


def get_args():
    parser = argparse.ArgumentParser(description="Generate StyleTTS2 synthetic speech for SAP speakers")

    parser.add_argument(
        "--sap-data-dir",
        type=Path,
        required=True,
        help="Path to sap/data dir (contains extracted/ and synthetic/)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="TRAIN",
        help="Dataset split to use (e.g. TRAIN, DEV, TEST)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["etiology", "speaker", "all"],
        required=True,
        help="Synthesis mode"
    )
    parser.add_argument(
        "--etiology",
        type=str,
        default=None,
        help="Etiology name (required for --mode etiology)"
    )
    parser.add_argument(
        "--speaker-id",
        type=str,
        default=None,
        help="Speaker ID (required for --mode speaker)"
    )
    parser.add_argument(
        "--ratings-csv",
        type=Path,
        default=None,
        help="Path to speaker_ratings CSV. Required for --mode etiology or --mode all."
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=10,
        help="StyleTTS2 diffusion steps (higher = better quality, slower)"
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Path to output speaker pairs CSV. Defaults to sap-data-dir/synthetic/speaker_pairs.csv"
    )

    return parser.parse_args()


@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


def load_styletts2():
    try:
        from styletts2 import tts
    except ImportError:
        logger.error("StyleTTS2 not installed. Run: pip install styletts2")
        return None

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading StyleTTS2 model (device: {device})...")
    if device == "cpu":
        logger.warning("CUDA not available — inference will be slow on CPU")

    with suppress_stdout_stderr():
        model = tts.StyleTTS2()

    if device == "cuda":
        try:
            model.model = model.model.to(device)
            logger.info(f"✓ StyleTTS2 loaded on {device}")
        except AttributeError:
            logger.warning("model.model attribute not found — checking available attributes:")
            logger.warning([a for a in dir(model) if not a.startswith('_')])
            logger.warning("Falling back to CPU")
    else:
        logger.info(f"✓ StyleTTS2 loaded on {device}")
    return model


def load_speaker_data(speaker_dir):
    """
    Load all utterances from a speaker's JSON file.
    Returns the first valid audio file as the reference clip,
    and a list of dicts with audio_path and transcript.
    """
    speaker_dir = Path(speaker_dir)
    json_files = list(speaker_dir.glob("*.json"))
    if not json_files:
        return None, []

    with open(json_files[0], 'r') as f:
        data = json.load(f)

    utterances = []
    for file_entry in data.get('Files', []):
        filename = file_entry.get('Filename', '')
        transcript = file_entry.get('Prompt', {}).get('Transcript', '')

        if not filename or not transcript:
            continue

        audio_path = speaker_dir / filename
        if audio_path.exists():
            utterances.append({
                'audio_path': audio_path,
                'transcript': transcript
            })

    if not utterances:
        return None, []

    # First utterance as reference clip
    reference_audio = utterances[0]['audio_path']
    return reference_audio, utterances


def get_speakers_for_etiology(extracted_dir, ratings_csv, etiology):
    """Return list of speaker IDs for a given etiology from ratings CSV."""
    df = pd.read_csv(ratings_csv)
    speaker_ids = df[df['Etiology'] == etiology]['Speaker_ID'].tolist()
    valid = []
    for s in tqdm(speaker_ids, desc=f"Scanning {etiology}", unit="speaker"):
        if (extracted_dir / s).exists():
            valid.append(s)
    logger.info(f"{etiology}: {len(valid)}/{len(speaker_ids)} speakers found in extracted dir")
    return valid, etiology


def synthesize_speaker(speaker_id, etiology, extracted_dir, synthetic_dir, tts_model, diffusion_steps):
    """
    Synthesize all utterances for one speaker.
    Returns list of pair dicts for the CSV.
    """
    speaker_dir = extracted_dir / speaker_id
    speaker_synthetic_dir = synthetic_dir / etiology.replace(" ", "_").replace("'", "") / speaker_id
    speaker_synthetic_dir.mkdir(parents=True, exist_ok=True)

    reference_audio, utterances = load_speaker_data(speaker_dir)
    if not utterances:
        logger.warning(f"No valid utterances for {speaker_id}, skipping.")
        return []

    pairs = []
    skipped = 0

    for utt in tqdm(utterances, desc=f"  {speaker_id}", leave=False, unit="utt"):
        utt_stem = utt['audio_path'].stem
        synthetic_path = speaker_synthetic_dir / f"{utt_stem}_synthetic.wav"

        # Skip already-synthesized
        if synthetic_path.exists():
            skipped += 1
            pairs.append({
                'speaker_id': speaker_id,
                'etiology': etiology,
                'original_audio': str(utt['audio_path']),
                'synthetic_audio': str(synthetic_path),
                'transcript': utt['transcript'],
                'status': 'skipped'
            })
            continue

        try:
            with suppress_stdout_stderr():
                tts_model.inference(
                    utt['transcript'],
                    target_voice_path=str(reference_audio),
                    output_wav_file=str(synthetic_path),
                    diffusion_steps=diffusion_steps,
                    alpha=0.3,
                    beta=0.7,
                    embedding_scale=1
                )
            pairs.append({
                'speaker_id': speaker_id,
                'etiology': etiology,
                'original_audio': str(utt['audio_path']),
                'synthetic_audio': str(synthetic_path),
                'transcript': utt['transcript'],
                'status': 'success'
            })
        except Exception as e:
            logger.error(f"  Failed on {utt_stem}: {e}")
            pairs.append({
                'speaker_id': speaker_id,
                'etiology': etiology,
                'original_audio': str(utt['audio_path']),
                'synthetic_audio': None,
                'transcript': utt['transcript'],
                'status': f'failed: {e}'
            })

    synthesized = sum(1 for p in pairs if p['status'] == 'success')
    logger.info(f"  {speaker_id}: {synthesized} synthesized, {skipped} skipped, "
                f"{sum(1 for p in pairs if 'failed' in p['status'])} failed")

    return pairs


def main():
    args = get_args()

    extracted_dir = args.sap_data_dir / "extracted" / args.split
    synthetic_dir = args.sap_data_dir / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    output_csv = args.output_csv or synthetic_dir / "speaker_pairs.csv"

    # Validate args
    if args.mode == "etiology" and not args.etiology:
        raise ValueError("--etiology is required when --mode etiology")
    if args.mode == "speaker" and not args.speaker_id:
        raise ValueError("--speaker-id is required when --mode speaker")
    if args.mode in ("etiology", "all") and not args.ratings_csv:
        raise ValueError("--ratings-csv is required for --mode etiology or --mode all")

    logger.info("=" * 60)
    logger.info("StyleTTS2 Synthetic Speech Generation")
    logger.info(f"Mode: {args.mode}")
    logger.info("=" * 60)

    tts_model = load_styletts2()
    if not tts_model:
        return

    # Build list of (speaker_id, etiology) to process
    targets = []

    if args.mode == "speaker":
        etiology = "Unknown"
        if args.ratings_csv:
            df = pd.read_csv(args.ratings_csv)
            match = df[df['Speaker_ID'] == args.speaker_id]
            if not match.empty:
                etiology = match.iloc[0]['Etiology']
        targets = [(args.speaker_id, etiology)]

    elif args.mode == "etiology":
        speaker_ids, etiology = get_speakers_for_etiology(
            extracted_dir, args.ratings_csv, args.etiology
        )
        targets = [(s, etiology) for s in speaker_ids]

    elif args.mode == "all":
        df = pd.read_csv(args.ratings_csv)
        for etiology, group in df.groupby('Etiology'):
            for speaker_id in tqdm(group['Speaker_ID'], desc=f"Scanning {etiology}", unit="speaker"):
                if (extracted_dir / speaker_id).exists():
                    targets.append((speaker_id, etiology))

    logger.info(f"Total speakers to process: {len(targets)}")

    all_pairs = []

    for speaker_id, etiology in tqdm(targets, desc="Processing speakers", unit="speaker"):
        logger.info(f"Processing {speaker_id} ({etiology})")
        pairs = synthesize_speaker(
            speaker_id, etiology,
            extracted_dir, synthetic_dir,
            tts_model, args.diffusion_steps
        )
        all_pairs.extend(pairs)

    # Write CSV
    results_df = pd.DataFrame(all_pairs)
    results_df.to_csv(output_csv, index=False)

    successful = len(results_df[results_df['status'] == 'success'])
    skipped = len(results_df[results_df['status'] == 'skipped'])
    failed = len(results_df[results_df['status'].str.startswith('failed', na=False)])

    logger.info("=" * 60)
    logger.info("Synthesis Complete!")
    logger.info(f"  Successful : {successful}")
    logger.info(f"  Skipped    : {skipped}")
    logger.info(f"  Failed     : {failed}")
    logger.info(f"  CSV saved  : {output_csv}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()