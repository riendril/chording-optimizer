"""
Dataset loader for corpus generation.

This module handles downloading, caching, and accessing subsets of public datasets
to ensure consistent corpus generation even when APIs fail.
"""

import gzip
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("dataset_loader")

# Dataset metadata - each entry contains information needed to fetch and process the data
DATASET_REGISTRY = {
    "general": [
        {
            "name": "wikitext_subset",
            "url": "https://huggingface.co/datasets/wikitext/resolve/main/wikitext-103-v1/wikitext-103-raw-v1-train.txt.gz?download=true",
            "file_size": 181_546_705,  # ~181MB
            "sample_size": 100_000,  # Characters to sample
            "max_entries": 50,  # Maximum entries to extract
            "format": "text",
            "desc": "Clean, wikipedia text samples",
        },
        {
            "name": "bookcorpus_subset",
            "url": "https://huggingface.co/datasets/bookcorpus/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet?download=true",
            "file_size": 104_857_600,  # ~100MB (approx)
            "sample_size": 150_000,  # Characters to sample
            "max_entries": 50,  # Maximum entries to extract
            "format": "parquet",
            "desc": "Book samples from various genres",
        },
    ],
    "scientific": [
        {
            "name": "arxiv_papers_subset",
            "url": "https://huggingface.co/datasets/ccdv/arxiv-classification/resolve/main/arxiv-filtered.jsonl?download=true",
            "file_size": 1_048_576,  # ~1MB (first chunk)
            "sample_size": 100_000,  # Characters to sample
            "max_entries": 100,  # Maximum entries to extract
            "format": "jsonl",
            "desc": "arXiv papers with abstracts and content",
        }
    ],
    "programming": [
        {
            "name": "github_readme_subset",
            "url": "https://huggingface.co/datasets/codeparrot/github-code/resolve/refs%2Fconvert%2Fparquet/default/train/0.parquet?download=true",
            "file_size": 2_097_152,  # ~2MB (first chunk)
            "sample_size": 80_000,  # Characters to sample
            "max_entries": 100,  # Maximum entries to extract
            "format": "parquet",
            "desc": "README files and code documentation from GitHub",
        }
    ],
    "forum": [
        {
            "name": "stackexchange_subset",
            "url": "https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_xml/resolve/refs%2Fconvert%2Fparquet/stackexchange_paired/train/0.parquet?download=true",
            "file_size": 2_097_152,  # ~2MB (first chunk)
            "sample_size": 80_000,  # Characters to sample
            "max_entries": 100,  # Maximum entries to extract
            "format": "parquet",
            "desc": "Discussions from StackExchange sites",
        }
    ],
    "subtitles": [
        {
            "name": "open_subtitles_subset",
            "url": "https://huggingface.co/datasets/open_subtitles/resolve/refs%2Fconvert%2Fjsonl/en/train/0.jsonl?download=true",
            "file_size": 2_097_152,  # ~2MB (first chunk)
            "sample_size": 50_000,  # Characters to sample
            "max_entries": 100,  # Maximum entries to extract
            "format": "jsonl",
            "desc": "Movie and TV subtitles",
        }
    ],
}


class DatasetLoader:
    """Handles downloading, caching and loading dataset subsets."""

    def __init__(self, cache_dir: Path):
        """Initialize the dataset loader.

        Args:
            cache_dir: Directory to store cached datasets
        """
        self.cache_dir = cache_dir / "datasets"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Track datasets that have been processed in this session
        self.processed_datasets = set()

    def download_file(
        self, url: str, target_path: Path, expected_size: Optional[int] = None
    ) -> bool:
        """Download a file with progress reporting.

        Args:
            url: URL to download
            target_path: Where to save the file
            expected_size: Expected file size in bytes (for progress bar)

        Returns:
            bool: True if download succeeded, False otherwise
        """
        try:
            # Stream download to handle large files efficiently
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Get file size for progress bar
            file_size = int(response.headers.get("content-length", 0))
            if expected_size and file_size > 0 and file_size > expected_size * 1.5:
                logger.warning(
                    f"File size ({file_size} bytes) is much larger than expected ({expected_size} bytes). Aborting download."
                )
                return False

            # Use file size from headers if available, otherwise use expected_size
            size_for_progress = (
                file_size
                if file_size > 0
                else (expected_size if expected_size else None)
            )

            # Show download progress
            desc = f"Downloading {target_path.name}"
            with tqdm(
                total=size_for_progress, unit="B", unit_scale=True, desc=desc
            ) as pbar:
                with open(target_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            # Verify the download was successful
            actual_size = target_path.stat().st_size
            if expected_size and actual_size < expected_size * 0.8:
                logger.warning(
                    f"Downloaded file size ({actual_size} bytes) is much smaller than expected ({expected_size} bytes)."
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            if target_path.exists():
                target_path.unlink()  # Remove partial download
            return False

    def get_dataset_path(self, dataset_info: Dict) -> Path:
        """Get the path where the dataset should be stored."""
        # Create a filename from the dataset name and a hash of the URL
        url_hash = hashlib.md5(dataset_info["url"].encode()).hexdigest()[:8]
        filename = f"{dataset_info['name']}_{url_hash}.json.gz"
        return self.cache_dir / filename

    def load_dataset(self, category: str, force_refresh: bool = False) -> List[str]:
        """Load dataset samples for a category.

        Args:
            category: Category to load (general, scientific, programming, etc.)
            force_refresh: Whether to force re-download and processing

        Returns:
            List of text samples from the dataset
        """
        if category not in DATASET_REGISTRY:
            logger.warning(f"No registered datasets for category: {category}")
            return []

        # Get dataset options for this category
        dataset_options = DATASET_REGISTRY[category]

        # If we have multiple dataset options, choose one randomly
        # This gives variety across corpus generation runs
        dataset_info = random.choice(dataset_options)

        # Check if we've already processed this dataset in this session
        dataset_key = f"{category}_{dataset_info['name']}"
        if dataset_key in self.processed_datasets:
            logger.info(
                f"Already processed {dataset_info['name']} in this session, skipping"
            )
            return []

        # Mark as processed to avoid reprocessing in the same session
        self.processed_datasets.add(dataset_key)

        # Get dataset path
        dataset_path = self.get_dataset_path(dataset_info)

        # Check if dataset already exists and we're not forcing a refresh
        if dataset_path.exists() and not force_refresh:
            try:
                with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
                    samples = json.load(f)
                    logger.info(
                        f"Loaded {len(samples)} samples from cached dataset {dataset_info['name']}"
                    )
                    return samples
            except Exception as e:
                logger.warning(
                    f"Error loading cached dataset {dataset_info['name']}: {e}"
                )
                # Will continue to download/process

        # Download the dataset if needed
        raw_dataset_path = self.cache_dir / f"{dataset_info['name']}_raw"

        download_needed = True
        if raw_dataset_path.exists() and not force_refresh:
            # Check if existing file seems valid
            if raw_dataset_path.stat().st_size > 1000:  # Arbitrary minimum size
                download_needed = False

        if download_needed:
            logger.info(f"Downloading dataset: {dataset_info['name']}")
            success = self.download_file(
                dataset_info["url"], raw_dataset_path, dataset_info["file_size"]
            )

            if not success:
                logger.error(f"Failed to download dataset: {dataset_info['name']}")
                # Try to use existing processed dataset if available
                if dataset_path.exists():
                    try:
                        with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
                            samples = json.load(f)
                            logger.info(
                                f"Using existing processed dataset despite download failure"
                            )
                            return samples
                    except Exception:
                        pass
                return []
        else:
            logger.info(f"Using existing raw dataset file for {dataset_info['name']}")

        # Process the dataset based on its format
        try:
            if dataset_info["format"] == "text":
                samples = self._process_text_dataset(
                    raw_dataset_path,
                    dataset_info["sample_size"],
                    dataset_info["max_entries"],
                )
            elif dataset_info["format"] == "jsonl":
                samples = self._process_jsonl_dataset(
                    raw_dataset_path,
                    dataset_info["sample_size"],
                    dataset_info["max_entries"],
                )
            elif dataset_info["format"] == "parquet":
                samples = self._process_parquet_dataset(
                    raw_dataset_path,
                    dataset_info["sample_size"],
                    dataset_info["max_entries"],
                )
            else:
                logger.error(f"Unknown dataset format: {dataset_info['format']}")
                return []

            # Cache the processed samples
            with gzip.open(dataset_path, "wt", encoding="utf-8") as f:
                json.dump(samples, f)

            logger.info(f"Processed {len(samples)} samples from {dataset_info['name']}")
            return samples

        except Exception as e:
            logger.error(f"Error processing dataset {dataset_info['name']}: {e}")
            import traceback

            traceback.print_exc()
            return []
        finally:
            # Clean up raw dataset file to save space
            if raw_dataset_path.exists():
                raw_dataset_path.unlink()

    def _process_text_dataset(
        self, file_path: Path, sample_size: int, max_entries: int
    ) -> List[str]:
        """Process a plain text dataset.

        Args:
            file_path: Path to the text file
            sample_size: Maximum size of each sample in characters
            max_entries: Maximum number of samples to extract

        Returns:
            List of text samples
        """
        samples = []

        # Check if it's a gzipped file
        open_func = gzip.open if str(file_path).endswith(".gz") else open
        mode = "rt" if str(file_path).endswith(".gz") else "r"

        with open_func(file_path, mode, encoding="utf-8") as f:
            # Read the file in chunks to avoid loading everything into memory
            buffer = ""
            total_entries = 0

            for line in f:
                # Add line to buffer
                buffer += line

                # If we have enough content, extract a sample
                if len(buffer) >= sample_size or "\n\n" in buffer:
                    # Look for paragraph breaks
                    parts = buffer.split("\n\n")

                    # If we have multiple paragraphs, process all but the last one
                    # (which might be incomplete)
                    for part in parts[:-1]:
                        if part.strip():  # Skip empty paragraphs
                            samples.append(part.strip())
                            total_entries += 1

                            if total_entries >= max_entries:
                                return samples

                    # Keep the last paragraph in the buffer
                    buffer = parts[-1]

                    # If buffer is larger than sample_size, extract a sample
                    if len(buffer) >= sample_size:
                        samples.append(buffer[:sample_size].strip())
                        buffer = buffer[sample_size:]
                        total_entries += 1

                        if total_entries >= max_entries:
                            return samples

        # Add any remaining content
        if buffer.strip():
            samples.append(buffer.strip())

        return samples

    def _process_jsonl_dataset(
        self, file_path: Path, sample_size: int, max_entries: int
    ) -> List[str]:
        """Process a JSONL dataset.

        Args:
            file_path: Path to the JSONL file
            sample_size: Maximum size of each sample in characters
            max_entries: Maximum number of samples to extract

        Returns:
            List of text samples
        """
        samples = []
        total_entries = 0

        # Check if it's a gzipped file
        open_func = gzip.open if str(file_path).endswith(".gz") else open
        mode = "rt" if str(file_path).endswith(".gz") else "r"

        with open_func(file_path, mode, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    # Parse JSON
                    entry = json.loads(line)

                    # Look for text fields in the JSON object
                    text = None

                    # Common text fields in datasets
                    for field in [
                        "text",
                        "content",
                        "abstract",
                        "body",
                        "document",
                        "sentence",
                        "description",
                    ]:
                        if (
                            field in entry
                            and isinstance(entry[field], str)
                            and len(entry[field]) > 100
                        ):
                            text = entry[field].strip()
                            break

                    # If we have text, add it as a sample
                    if text:
                        # Limit to sample_size
                        if len(text) > sample_size:
                            # Try to find a paragraph break near sample_size
                            cutoff = min(len(text), sample_size)
                            paragraph_break = text.rfind("\n\n", 0, cutoff)
                            if paragraph_break > 0:
                                text = text[:paragraph_break].strip()
                            else:
                                text = text[:cutoff].strip()

                        samples.append(text)
                        total_entries += 1

                        if total_entries >= max_entries:
                            break

                except json.JSONDecodeError:
                    continue

        return samples

    def _process_parquet_dataset(
        self, file_path: Path, sample_size: int, max_entries: int
    ) -> List[str]:
        """Process a Parquet dataset.

        Args:
            file_path: Path to the Parquet file
            sample_size: Maximum size of each sample in characters
            max_entries: Maximum number of samples to extract

        Returns:
            List of text samples
        """
        # Import pyarrow only when needed
        try:
            import pyarrow.parquet as pq
        except ImportError:
            logger.error(
                "pyarrow is required to process Parquet files. Please install it with pip install pyarrow"
            )
            return []

        samples = []

        try:
            # Read the Parquet file
            table = pq.read_table(file_path)

            # Convert to pandas for easier processing
            import pandas as pd

            df = table.to_pandas()

            # Look for text columns
            text_columns = []
            for col in df.columns:
                # Check if column contains strings
                if df[col].dtype == "object":
                    # Check a sample to see if it contains text
                    sample = df[col].iloc[0] if not df[col].empty else None
                    if isinstance(sample, str) and len(sample) > 50:
                        text_columns.append(col)

            # Process text columns
            total_entries = 0

            for col in text_columns:
                for text in df[col]:
                    if not isinstance(text, str) or len(text) < 100:
                        continue

                    # Limit to sample_size
                    if len(text) > sample_size:
                        # Try to find a paragraph break near sample_size
                        cutoff = min(len(text), sample_size)
                        paragraph_break = text.rfind("\n\n", 0, cutoff)
                        if paragraph_break > 0:
                            text = text[:paragraph_break].strip()
                        else:
                            text = text[:cutoff].strip()

                    samples.append(text.strip())
                    total_entries += 1

                    if total_entries >= max_entries:
                        return samples

        except Exception as e:
            logger.error(f"Error processing Parquet file: {e}")

        return samples

    def get_available_dataset_info(self) -> Dict[str, List[Dict]]:
        """Get information about available datasets for each category."""
        return DATASET_REGISTRY.copy()

    def cleanup_cache(self, max_age_days: int = 30) -> None:
        """Clean up old cached datasets.

        Args:
            max_age_days: Maximum age of cached files in days
        """
        import time

        now = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60

        for file_path in self.cache_dir.glob("*"):
            if file_path.is_file():
                file_age = now - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        logger.info(f"Removed old cached dataset: {file_path.name}")
                    except Exception as e:
                        logger.warning(
                            f"Error removing cached file {file_path.name}: {e}"
                        )
