#!/usr/bin/env python3
"""
Script to process all CAAML files in data/snowpits directory and identify
which ones contain PST (Propagation Saw Test) data.
"""

import os
from pathlib import Path
from snowpylot import SnowPit
from weac_2.utils.snowpilot_parser import SnowPilotParser
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_all_caaml_files(base_dir):
    """Find all CAAML files in the snowpits directory structure."""
    caaml_files = []
    base_path = Path(base_dir)

    if not base_path.exists():
        logger.error("Directory %s does not exist", base_dir)
        return []

    # Look for .xml files (CAAML format) in all subdirectories
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith((".xml", ".caaml")):
                file_path = Path(root) / file
                caaml_files.append(file_path)

    logger.info("Found %d CAAML files", len(caaml_files))
    return caaml_files


def check_for_pst_data(snowpit: SnowPit):
    """
    Check if any of the model inputs contain PST data.
    PST data would be indicated by specific stability test results.
    """
    if not snowpit:
        return False

    return len(snowpit.stability_tests.PST) > 0


def process_caaml_files():
    """Process all CAAML files and identify those with PST data."""
    base_dir = "data/snowpits"
    caaml_files = find_all_caaml_files(base_dir)

    if not caaml_files:
        logger.warning("No CAAML files found in %s", base_dir)
        return

    pst_files = []
    error_files = []
    processed_count = 0

    logger.info("Processing %d CAAML files...", len(caaml_files))

    for file_path in caaml_files:
        try:
            logger.debug("Processing file: %s", file_path)

            # Create parser and process the file
            snowpit_parser = SnowPilotParser(str(file_path))

            # Check if this file contains PST data
            if check_for_pst_data(snowpit_parser.snowpit):
                pst_files.append(file_path)
                logger.info("PST found in: %s", file_path.name)

            processed_count += 1

            # Progress update every 50 files
            if processed_count % 50 == 0:
                logger.info("Processed %d/%d files", processed_count, len(caaml_files))

        except Exception as e:
            logger.error("Error processing %s: %s", file_path.name, str(e))
            error_files.append((file_path, str(e)))

    # Summary
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info("Total files processed: %d", processed_count)
    logger.info("Files with PST data: %d", len(pst_files))
    logger.info("Files with errors: %d", len(error_files))

    # if pst_files:
    #     logger.info("\nFiles containing PST data:")
    #     for pst_file in pst_files:
    #         # Show relative path from the base directory
    #         relative_path = pst_file.relative_to(Path(base_dir))
    #         logger.info("  - %s", relative_path)

    # if error_files:
    #     logger.info("\nFiles with processing errors:")
    #     for error_file, error_msg in error_files[:10]:  # Show first 10 errors
    #         relative_path = error_file.relative_to(Path(base_dir))
    #         logger.info("  - %s: %s", relative_path, error_msg)
    #     if len(error_files) > 10:
    #         logger.info("  ... and %d more errors", len(error_files) - 10)

    return pst_files, error_files


if __name__ == "__main__":
    pst_files, error_files = process_caaml_files()
