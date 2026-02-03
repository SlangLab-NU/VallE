import logging
import tarfile
import os
import sys
import json
import io
from collections import defaultdict
from tqdm.auto import tqdm
from pathlib import Path

DATASET_DIR = Path("extracted/")
subfolders = ["DEV", "TRAIN"]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def count_speakers(data_dir):
    """
    Counts the total speakers within the dev and train sets
    """
    dev_count = set()
    train_count = set()
    dev_paths = [] 
    train_paths = []  
    
    for folder in subfolders:
        folder_path = data_dir / folder
        speaker_dirs = [item for item in folder_path.iterdir() if item.is_dir()]
        
        if folder == "DEV":
            dev_count.update([item.name for item in speaker_dirs])
            dev_paths.extend(speaker_dirs)
        else:
            train_count.update([item.name for item in speaker_dirs])
            train_paths.extend(speaker_dirs)
    
    # Calculate averages
    dev_avg = count_average_utterances(dev_paths)
    train_avg = count_average_utterances(train_paths)
    
    logger.info("\nSPEAKER COUNTS\n"
                "---------------\n"
                f"dev count: {len(dev_count)}\n"
                f"train count: {len(train_count)}\n"
                f"Total speakers: {len(dev_count) + len(train_count)}\n"
                "\nAVERAGE UTTERANCES\n"
                "---------------------\n"
                f"Average Utterances Dev: {dev_avg:.2f}\n"
                f"Average Utterances Train: {train_avg:.2f}")


def count_average_utterances(speaker_paths):
    """
    speaker_paths: list of Path objects pointing to speaker directories
    """
    num_speakers = len(speaker_paths)
    if num_speakers == 0:
        return 0
    
    utterance_count = 0
    for speaker_dir in speaker_paths:
        wav_files = list(speaker_dir.glob("*.wav"))
        json_files = list(speaker_dir.glob("*.json"))
        utterance_count += len(wav_files)
    
    return utterance_count / num_speakers


def analyze_speaker_categories(speaker_dir):
    """
    Analyze utterances per category for a single speaker.
    Returns dict with category counts and speaker metadata.
    """
    # Find the JSON file
    json_files = list(speaker_dir.glob("*.json"))
    if not json_files:
        logger.warning(f"No JSON found for {speaker_dir.name}")
        return None
    
    json_path = json_files[0]
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    contributor_id = data.get("Contributor ID", "Unknown")
    etiology = data.get("Etiology", "Unknown")
    
    # Count utterances per category
    category_counts = defaultdict(int)
    total_utterances = 0
    
    for file_entry in data.get("Files", []):
        total_utterances += 1
        prompt = file_entry.get("Prompt", {})
        category = prompt.get("Category Description", "Unknown")
        category_counts[category] += 1
    
    return {
        'contributor_id': contributor_id,
        'etiology': etiology,
        'total_utterances': total_utterances,
        'category_counts': dict(category_counts)
    }


def analyze_dataset_categories(data_dir, dataset_type):
    """
    Analyze all speakers in a dataset and compute per-speaker category stats.
    """
    folder_path = data_dir / dataset_type
    speaker_dirs = [item for item in folder_path.iterdir() if item.is_dir()]
    
    all_speaker_stats = []
    
    logger.info(f"\nAnalyzing {dataset_type} categories...")
    
    for speaker_dir in tqdm(speaker_dirs, desc=f"{dataset_type} speakers"):
        stats = analyze_speaker_categories(speaker_dir)
        if stats:
            all_speaker_stats.append(stats)
    
    # Compute averages across all speakers
    if not all_speaker_stats:
        logger.warning(f"No data found for {dataset_type}")
        return
    
    # Get all unique categories
    all_categories = set()
    for stats in all_speaker_stats:
        all_categories.update(stats['category_counts'].keys())
    
    # Calculate average utterances per category across all speakers
    category_totals = defaultdict(int)
    category_speaker_counts = defaultdict(int)
    
    for stats in all_speaker_stats:
        for category, count in stats['category_counts'].items():
            category_totals[category] += count
            category_speaker_counts[category] += 1

    message_parts = [
        f"\n{dataset_type} CATEGORY ANALYSIS",
        "=" * 60,
        f"Total speakers: {len(all_speaker_stats)}",
        "\nCategory breakdown:"
]
    
    for category in sorted(all_categories):
        total = category_totals[category]
        num_speakers = category_speaker_counts[category]
        avg_per_speaker = total / num_speakers if num_speakers > 0 else 0
        
        message_parts.extend([
            f"  {category}:",
            f"    Total utterances: {total}",
            f"    Speakers with this category: {num_speakers}",
            f"    Avg per speaker: {avg_per_speaker:.2f}"
        ])
    logger.info("\n".join(message_parts))
    return all_speaker_stats


def count_speakers_with_categories(data_dir):
    """
    Main function to analyze both DEV and TRAIN datasets.
    """
    logger.info("\nCATEGORY ANALYSIS")
    logger.info("=" * 60)

    dev_stats = analyze_dataset_categories(data_dir, "DEV")
    train_stats = analyze_dataset_categories(data_dir, "TRAIN")
    
    return dev_stats, train_stats


def main():
    logger.info("Starting SAP dataset analysis")
    
    count_speakers(DATASET_DIR)
    
    count_speakers_with_categories(DATASET_DIR)


if __name__ == "__main__":
    main()