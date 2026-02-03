
import tarfile
import os
import sys
import json
import io
from pathlib import Path
from collections import defaultdict
import argparse
from tqdm.auto import tqdm
import logging

DEFAULT_DIR = Path("/scratch/lewis.jor/VallE/egs/sap/download")
OUTPUT_DIR = Path("/scratch/lewis.jor/VallE/egs/sap/")
subfolders = ["DEV", "TRAIN"]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Redirect logger output to work with tqdm
class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)  # Write to tqdm's output stream
        except Exception:
            self.handleError(record)

# Apply the handler
logger.handlers = [TqdmLoggingHandler()]
logger = logging.getLogger(__name__)


def is_already_extracted(tar_path, extract_to):
    """
    Check if tar file has already been extracted by looking at its contents
    """
    try:
        with tarfile.open(tar_path, 'r') as tar:
            # Get first member to check
            members = tar.getmembers()
            if not members:
                return False

            # Check if first file/folder from tar exists in extract location
            first_item = members[0].name
            expected_path = extract_to / first_item

            return expected_path.exists()
    except Exception:
        return False


def extract_sap(download_dir, output_dir, subfolders):
    """
    Extract the downloaded SAP to a new extracted dir
    """
    extracted_dir = output_dir / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    dev_dir = extracted_dir / "DEV"
    train_dir = extracted_dir / "TRAIN"
    dev_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting extraction from: {download_dir}")
    
    checkpoint_file = extracted_dir / ".outer_extraction.done"
        
    for subfolder_name in tqdm(subfolders, desc="Extracting folder contents", position=0):
        if not checkpoint_file.exists():
            subfolder = download_dir / subfolder_name

            if not subfolder.exists():
                logger.warning(f"Subfolder not found: {subfolder}")
                continue
            
            if subfolder_name == "DEV":
                extract_to = dev_dir
            else:
                extract_to = train_dir

            tar_files = list(subfolder.glob("*.tar"))
            logger.info(f"Found {len(tar_files)} files in {subfolder_name}")
        
            # Inner progress bar for tar files
            for tar_path in tqdm(tar_files, desc=f"{subfolder_name}", position=1, leave=False):
                try:
                    if is_already_extracted(tar_path, extract_to):
                        logger.debug(f"Skipping already extracted: {tar_path.name}")
                        continue
                    with tarfile.open(tar_path, 'r') as tar:
                        tar.extractall(path=extract_to)
            
                except Exception as e:  
                    logger.error(f"Failed: {tar_path.name} - {e}")

    logger.info("Searching for nested tar files...")
    for target_dir in [dev_dir, train_dir]:
        nested_tars = list(target_dir.rglob("*.tar"))  # Recursive search
        
        if nested_tars:
            logger.info(f"Found {len(nested_tars)} nested tar files in {target_dir.name}")
            
            for nested_tar in tqdm(nested_tars, desc=f"Nested {target_dir.name}", position=0):
                try:
                    # Extract in the same directory as the nested tar
                    extract_location = nested_tar.parent
                    
                    if is_already_extracted(nested_tar, extract_location):
                        logger.debug(f"Skipping already extracted nested: {nested_tar.name}")
                        continue

                    with tarfile.open(nested_tar, 'r') as tar:
                        tar.extractall(path=extract_location)
                    
                    # Optionally remove the nested tar after extraction
                    nested_tar.unlink()
                    logger.info(f"Extracted and removed: {nested_tar.name}")
                    
                except Exception as e:
                    logger.error(f"Failed nested extraction: {nested_tar.name} - {e}")

    logger.info("Extraction complete")


def main():

    extract_sap(DEFAULT_DIR, OUTPUT_DIR, subfolders)


if __name__ == "__main__":
    main()
