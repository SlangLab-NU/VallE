import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import logging

# Setup paths
EXTRACTED_DIR = Path("/scratch/lewis.jor/VallE/egs/sap/extracted")
OUTPUT_DIR = Path("/scratch/lewis.jor/VallE/egs/sap/analysis/histograms")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_speaker_json(json_path):
    """Parse a speaker's JSON file and extract rating data."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    contributor_id = data.get("Contributor ID", "Unknown")
    etiology = data.get("Etiology", "Unknown")
    
    # Collect ratings per dimension
    dimension_ratings = defaultdict(list)
    total_utterances = 0
    unrated_utterances = 0
    
    for file_entry in data.get("Files", []):
        total_utterances += 1
        ratings = file_entry.get("Ratings", [])
        
        if not ratings:
            unrated_utterances += 1
            continue
        
        # Group ratings by dimension
        for rating in ratings:
            dimension = rating.get("Dimension Description", "Unknown")
            level = rating.get("Level")
            
            if level is not None:
                try:
                    dimension_ratings[dimension].append(int(level))
                except ValueError:
                    logger.warning(f"Invalid level '{level}' for dimension '{dimension}'")
    
    return {
        'contributor_id': contributor_id,
        'etiology': etiology,
        'dimension_ratings': dimension_ratings,
        'total_utterances': total_utterances,
        'unrated_utterances': unrated_utterances
    }


def create_histogram(speaker_data, output_path, dataset_type):
    """Create and save histogram for a speaker."""
    contributor_id = speaker_data['contributor_id']
    etiology = speaker_data['etiology']
    dimension_ratings = speaker_data['dimension_ratings']
    unrated = speaker_data['unrated_utterances']
    total = speaker_data['total_utterances']
    
    if not dimension_ratings:
        logger.warning(f"No ratings found for {contributor_id}, skipping histogram")
        return
    
    # Calculate averages
    dimensions = []
    averages = []
    
    for dimension, levels in sorted(dimension_ratings.items()):
        dimensions.append(dimension)
        averages.append(np.mean(levels))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create bar plot
    x_pos = np.arange(len(dimensions))
    bars = ax.bar(x_pos, averages, color='steelblue', alpha=0.8)
    
    # Customize plot
    ax.set_xlabel('Dimension', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Level', fontsize=12, fontweight='bold')
    ax.set_title(f'{contributor_id} - {etiology}\n({dataset_type} Set)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(dimensions, rotation=45, ha='right')
    ax.set_ylim(0, max(averages) * 1.1 if averages else 5)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, avg) in enumerate(zip(bars, averages)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{avg:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    # Add unrated utterances text
    plt.figtext(0.99, 0.01, 
                f'Total utterances: {total} | Unrated: {unrated}',
                ha='right', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved histogram: {output_path.name}")


def process_dataset(dataset_dir, dataset_type):
    """Process all speakers in a dataset (DEV or TRAIN)."""
    output_subdir = OUTPUT_DIR / dataset_type
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing {dataset_type} dataset from {dataset_dir}")
    
    # Find all speaker directories
    speaker_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    
    if not speaker_dirs:
        logger.warning(f"No speaker directories found in {dataset_dir}")
        return
    
    for speaker_dir in tqdm(speaker_dirs, desc=f"Processing {dataset_type} speakers"):
        # Find JSON file
        json_files = list(speaker_dir.glob("*.json"))
        
        if not json_files:
            logger.warning(f"No JSON file found in {speaker_dir.name}")
            continue
        
        if len(json_files) > 1:
            logger.warning(f"Multiple JSON files found in {speaker_dir.name}, using first")
        
        json_path = json_files[0]
        
        try:
            # Parse speaker data
            speaker_data = parse_speaker_json(json_path)
            
            # Create output filename
            contributor_id = speaker_data['contributor_id']
            output_filename = f"{contributor_id}_{dataset_type}.png"
            output_path = output_subdir / output_filename
            
            # Create histogram
            create_histogram(speaker_data, output_path, dataset_type)
            
        except Exception as e:
            logger.error(f"Failed to process {speaker_dir.name}: {e}")


def main():
    """Main processing function."""
    logger.info("Starting histogram generation")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process DEV dataset
    dev_dir = EXTRACTED_DIR / "DEV"
    if dev_dir.exists():
        process_dataset(dev_dir, "DEV")
    else:
        logger.warning(f"DEV directory not found: {dev_dir}")
    
    # Process TRAIN dataset
    train_dir = EXTRACTED_DIR / "TRAIN"
    if train_dir.exists():
        process_dataset(train_dir, "TRAIN")
    else:
        logger.warning(f"TRAIN directory not found: {train_dir}")
    
    logger.info(f"Histogram generation complete. Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
