import logging
import tarfile
import os
import sys
import json
import io
import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
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


def create_etiology_histogram(data_dir, dataset_type, output_dir):
    """
    Create histogram showing number of speakers per etiology.
    """
    folder_path = data_dir / dataset_type
    speaker_dirs = [item for item in folder_path.iterdir() if item.is_dir()]
    
    etiologies = []
    
    # Collect etiology for each speaker
    for speaker_dir in tqdm(speaker_dirs, desc=f"Collecting {dataset_type} etiologies"):
        json_files = list(speaker_dir.glob("*.json"))
        if not json_files:
            continue
        
        try:
            with open(json_files[0], 'r') as f:
                data = json.load(f)
                etiology = data.get("Etiology", "Unknown")
                etiologies.append(etiology)
        except Exception as e:
            logger.error(f"Failed to read {speaker_dir.name}: {e}")
    

    etiology_counts = Counter(etiologies)
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(12, 6))
    
    etiologies_sorted = sorted(etiology_counts.items(), key=lambda x: x[1], reverse=True)
    categories = [e[0] for e in etiologies_sorted]
    counts = [e[1] for e in etiologies_sorted]
    
    bars = ax.bar(categories, counts, color='steelblue', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Etiology', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Speakers', fontsize=12, fontweight='bold')
    ax.set_title(f'Speaker Distribution by Etiology - {dataset_type} Set', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    total = sum(counts)
    plt.figtext(0.99, 0.01, 
                f'Total speakers: {total}',
                ha='right', fontsize=10, style='italic')
    
    plt.tight_layout()
    
    output_path = output_dir / f"etiology_distribution_{dataset_type}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved etiology histogram: {output_path}")
    
    # Log breakdown
    logger.info(f"\n{dataset_type} ETIOLOGY DISTRIBUTION")
    logger.info("=" * 60)
    for etiology, count in etiologies_sorted:
        percentage = (count / total) * 100
        logger.info(f"  {etiology}: {count} ({percentage:.1f}%)")
    logger.info(f"  Total: {total}")
    
    return etiology_counts


def analyze_etiologies(data_dir):
    """
    Create etiology histograms for both DEV and TRAIN datasets.
    """
    output_dir = Path("/scratch/lewis.jor/VallE/egs/sap/analysis/etiologies")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\nETIOLOGY ANALYSIS")
    logger.info("=" * 60)
    
    # Analyze DEV
    dev_etiologies = create_etiology_histogram(data_dir, "DEV", output_dir)
    
    # Analyze TRAIN
    train_etiologies = create_etiology_histogram(data_dir, "TRAIN", output_dir)
    
    return dev_etiologies, train_etiologies


def analyze_speaker_ratings(data_dir, dataset_type, output_dir):
    """
    Calculate average ratings per speaker within each etiology.
    Export one row per speaker to CSV.
    """
    folder_path = data_dir / dataset_type
    speaker_dirs = [item for item in folder_path.iterdir() if item.is_dir()]
    
    csv_data = []
    
    logger.info(f"\nAnalyzing {dataset_type} speaker ratings...")
    
    for speaker_dir in tqdm(speaker_dirs, desc=f"Processing {dataset_type} speakers"):
        json_files = list(speaker_dir.glob("*.json"))
        if not json_files:
            continue
        
        try:
            with open(json_files[0], 'r') as f:
                data = json.load(f)
            
            speaker_id = data.get("Contributor ID", "Unknown")
            etiology = data.get("Etiology", "Unknown")
            
            all_ratings = []
            rated_utterances = 0
            unrated_utterances = 0
            
            for file_entry in data.get("Files", []):
                ratings = file_entry.get("Ratings", [])
                
                if not ratings:
                    unrated_utterances += 1
                else:
                    rated_utterances += 1
                    
                    for rating in ratings:
                        level = rating.get("Level")
                        if level is not None:
                            try:
                                all_ratings.append(int(level))
                            except ValueError:
                                continue
            
            if all_ratings:
                avg_rating = sum(all_ratings) / len(all_ratings)
            else:
                avg_rating = None
            
            total_utterances = rated_utterances + unrated_utterances
            
            csv_data.append({
                'Speaker_ID': speaker_id,
                'Etiology': etiology,
                'Average_Rating': round(avg_rating, 2) if avg_rating is not None else 'N/A',
                'Number_of_Ratings': rated_utterances,
                'Number_Not_Rated': unrated_utterances,
                'Total_Utterances': total_utterances,
                'Dataset': dataset_type  # Keep for combined CSV
            })
        
        except Exception as e:
            logger.error(f"Failed to process {speaker_dir.name}: {e}")
    
    # Export individual CSV (without Dataset column)
    output_path = output_dir / f"speaker_ratings_{dataset_type}.csv"
    
    if csv_data:
        fieldnames = ['Speaker_ID', 'Etiology', 'Average_Rating', 'Number_of_Ratings', 
                      'Number_Not_Rated', 'Total_Utterances']
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(csv_data)
        
        logger.info(f"Saved speaker ratings CSV: {output_path}")
    else:
        logger.warning(f"No rating data found for {dataset_type}")
    
    # Log summary by etiology
    etiology_groups = defaultdict(list)
    for row in csv_data:
        etiology_groups[row['Etiology']].append(row)
    
    message_parts = [
        f"\n{dataset_type} SPEAKER RATING SUMMARY BY ETIOLOGY",
        "=" * 60
    ]
    
    for etiology in sorted(etiology_groups.keys()):
        speakers = etiology_groups[etiology]
        avg_ratings = [float(s['Average_Rating']) for s in speakers if s['Average_Rating'] != 'N/A']
        
        message_parts.append(f"\n{etiology}:")
        message_parts.append(f"  Number of speakers: {len(speakers)}")
        if avg_ratings:
            message_parts.append(f"  Avg rating across speakers: {sum(avg_ratings)/len(avg_ratings):.2f}")
            message_parts.append(f"  Range: {min(avg_ratings):.2f} - {max(avg_ratings):.2f}")
    
    logger.info("\n".join(message_parts))
    
    return csv_data


def export_speaker_ratings(data_dir):
    """
    Analyze and export speaker ratings for both DEV and TRAIN.
    """
    output_dir = Path("analysis/ratings")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\nSPEAKER RATING ANALYSIS")
    logger.info("=" * 60)
    
    # Analyze DEV
    dev_ratings = analyze_speaker_ratings(data_dir, "DEV", output_dir)
    
    # Analyze TRAIN
    train_ratings = analyze_speaker_ratings(data_dir, "TRAIN", output_dir)
    
    # Create combined CSV (with Dataset column)
    combined_path = output_dir / "speaker_ratings_combined.csv"
    if dev_ratings or train_ratings:
        all_data = dev_ratings + train_ratings
        
        fieldnames = ['Speaker_ID', 'Etiology', 'Average_Rating', 'Number_of_Ratings', 
                      'Number_Not_Rated', 'Total_Utterances', 'Dataset']
        
        with open(combined_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_data)
        
        logger.info(f"Saved combined speaker ratings CSV: {combined_path}")
    
    return dev_ratings, train_ratings

def main():
    logger.info("Starting SAP dataset analysis")
    
    count_speakers(DATASET_DIR)
    
    count_speakers_with_categories(DATASET_DIR)

    analyze_etiologies(DATASET_DIR)
    export_speaker_ratings(DATASET_DIR)

if __name__ == "__main__":
    main()