"""
Enhanced Corpus Generator for Chord Layout Optimization

This script generates custom corpora by sampling from:
1. Public dataset subsets (like WikiText, arXiv, etc.)
2. Web API sources (when available)
3. Cache of previously generated content

It intelligently balances content from different sources to create a representative
corpus for chord layout analysis, with integrated fallbacks when APIs fail.
"""

import argparse
import gzip
import json
import logging
import os
import random
import re
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import requests

# Import the config
from src.common.config import GeneratorConfig

# Import the dataset loader
from src.token_generation.dataset_loader import DatasetLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("corpus_generator")

# -----------------
# Rate Limiting
# -----------------


class RateLimiter:
    """Simple rate limiter to avoid hitting API limits"""

    def __init__(self, calls_per_minute=30, max_retries=3):
        self.calls_per_minute = calls_per_minute
        self.call_timestamps = deque()
        self.max_retries = max_retries

    def wait_if_needed(self):
        """Wait if we've made too many calls recently"""
        now = datetime.now()

        # Remove timestamps older than a minute
        while (
            self.call_timestamps
            and (now - self.call_timestamps[0]).total_seconds() > 60
        ):
            self.call_timestamps.popleft()

        # If we've made too many calls, wait
        if len(self.call_timestamps) >= self.calls_per_minute:
            oldest_timestamp = self.call_timestamps[0]
            seconds_since_oldest = (now - oldest_timestamp).total_seconds()
            seconds_to_wait = max(0, 60 - seconds_since_oldest)

            if seconds_to_wait > 0:
                logger.info(f"Rate limiting: waiting {seconds_to_wait:.1f} seconds")
                time.sleep(seconds_to_wait)

        # Record this call
        self.call_timestamps.append(datetime.now())

    def call_with_retry(self, func, *args, **kwargs):
        """Call a function with retry logic and rate limiting"""
        retries = 0
        while retries <= self.max_retries:
            self.wait_if_needed()

            try:
                return func(*args, **kwargs)
            except requests.exceptions.RequestException as e:
                retries += 1
                if retries > self.max_retries:
                    raise

                # Exponential backoff
                wait_time = 2**retries
                logger.warning(
                    f"Request failed: {e}. Retrying in {wait_time} seconds..."
                )
                time.sleep(wait_time)

        raise Exception("Maximum retries exceeded")


# Global rate limiters for different sources
rate_limiters = {
    "reddit": RateLimiter(calls_per_minute=30),
    "stackexchange": RateLimiter(calls_per_minute=25),
    "arxiv": RateLimiter(calls_per_minute=20),
    "github": RateLimiter(calls_per_minute=10),
    "opensubtitles": RateLimiter(calls_per_minute=5),
    "default": RateLimiter(calls_per_minute=15),
}

# -----------------
# Data Source Configurations
# -----------------

SOURCES = {
    "forum": [
        {
            "name": "reddit_pushshift",
            "endpoint": "https://api.pushshift.io/reddit/search/submission",
            "params": {
                "sort": "score",
                "sort_type": "score",
                "size": 100,
                "is_self": True,
                "subreddit": "AskReddit,explainlikeimfive,TrueAskReddit,writing,CasualConversation,askscience,programming",
            },
            "extract_function": "extract_reddit_text",
            "rate_limiter": "reddit",
        },
        {
            "name": "stackexchange_general",
            "endpoint": "https://api.stackexchange.com/2.3/questions",
            "params": {
                "order": "desc",
                "sort": "votes",
                "site": "stackoverflow",
                "filter": "withbody",
                "pagesize": 100,
            },
            "extract_function": "extract_stackexchange_text",
            "rate_limiter": "stackexchange",
        },
    ],
    "subtitles": [
        {
            "name": "opensubtitles",
            "endpoint": "https://api.opensubtitles.com/api/v1/subtitles",
            "params": {
                "languages": "en",
                "order_by": "download_count",
                "page": 1,
            },
            "headers": {
                "Api-Key": "YOUR_OPENSUBTITLES_API_KEY",  # Will be replaced from config
                "Content-Type": "application/json",
            },
            "extract_function": "extract_opensubtitles_text",
            "rate_limiter": "opensubtitles",
        },
    ],
    "scientific": [
        {
            "name": "arxiv",
            "endpoint": "http://export.arxiv.org/api/query",
            "params": {
                "search_query": "all:physics OR all:cs OR all:math",
                "start": 0,
                "max_results": 100,
                "sortBy": "relevance",
            },
            "extract_function": "extract_arxiv_text",
            "rate_limiter": "arxiv",
        },
    ],
    "programming": [
        {
            "name": "stackexchange_programming",
            "endpoint": "https://api.stackexchange.com/2.3/questions",
            "params": {
                "order": "desc",
                "sort": "activity",
                "tagged": "python;javascript;java;c++;go",
                "site": "stackoverflow",
                "filter": "withbody",
                "pagesize": 100,
            },
            "extract_function": "extract_stackexchange_text",
            "rate_limiter": "stackexchange",
        },
        {
            "name": "github_readme",
            "endpoint": "https://api.github.com/search/repositories",
            "params": {
                "q": "stars:>1000",
                "sort": "stars",
                "order": "desc",
                "per_page": 50,
            },
            "extract_function": "extract_github_readme",
            "rate_limiter": "github",
        },
    ],
    "general": [
        {
            "name": "project_gutenberg",
            "endpoint": "https://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.bz2",
            "extract_function": "extract_gutenberg_text",
            "rate_limiter": "default",
            # Will use local caching instead
        },
    ],
}

USER_AGENT = "ChordCorpusGenerator/2.0"

# -----------------
# Extraction Functions
# -----------------


def extract_reddit_text(response_data: Dict) -> List[str]:
    """Extract text content from Reddit API response."""
    texts = []

    # Handle error responses
    if isinstance(response_data, dict) and "status_code" in response_data:
        return []

    for post in response_data.get("data", []):
        title = post.get("title", "")
        selftext = post.get("selftext", "")

        if len(selftext) > 100:  # Only use posts with substantial content
            # Combine title and content
            full_text = f"{title}\n\n{selftext}"

            # Clean text
            text = clean_text(full_text)
            if text:  # Only add non-empty texts
                texts.append(text)

    return texts


def extract_stackexchange_text(response_data: Dict) -> List[str]:
    """Extract text content from Stack Exchange API response."""
    texts = []

    # Handle error responses
    if isinstance(response_data, dict) and "status_code" in response_data:
        return []

    for item in response_data.get("items", []):
        title = item.get("title", "")
        body = item.get("body", "")

        # Get answers if they exist
        answers = []
        if "answers" in item:
            for answer in item.get("answers", []):
                answers.append(answer.get("body", ""))

        # Clean HTML from body
        body = re.sub(r"<[^>]+>", " ", body)

        if len(body) > 100:  # Only use posts with substantial content
            # Build the full text
            full_text = f"{title}\n\n{body}"

            # Add answers if they exist
            if answers:
                # Add the top answer
                answer_text = re.sub(r"<[^>]+>", " ", answers[0])
                full_text += f"\n\nTop Answer:\n{answer_text}"

            # Clean text
            text = clean_text(full_text)
            if text:  # Only add non-empty texts
                texts.append(text)

    return texts


def extract_opensubtitles_text(response_data: Dict) -> List[str]:
    """Extract text content from OpenSubtitles API response."""
    texts = []

    # If API key is not configured or API fails, rely on cached data only
    if isinstance(response_data, dict) and "status_code" in response_data:
        logger.warning(
            "OpenSubtitles API access failed. Will use cached data only if available."
        )
        return []

    for subtitle_data in response_data.get("data", []):
        file_id = subtitle_data.get("id")
        content = subtitle_data.get("content", "")

        if content:
            # Clean text
            text = clean_text(content)
            if text:
                texts.append(text)

    return texts


def extract_arxiv_text(response_data: str) -> List[str]:
    """Extract text content from arXiv API response (XML)."""
    texts = []

    # Handle error responses
    if isinstance(response_data, dict) and "status_code" in response_data:
        return []

    # If it's a string, process as XML
    if not isinstance(response_data, str):
        response_data = response_data.get("text", "")

    # Extract titles and abstracts using regex
    # In a production environment, you'd want to use an XML parser
    titles = re.findall(r"<title>(.*?)</title>", response_data, re.DOTALL)
    abstracts = re.findall(r"<summary>(.*?)</summary>", response_data, re.DOTALL)

    # Remove the first title (feed title)
    if titles:
        titles = titles[1:]

    # Combine titles with abstracts
    for i in range(min(len(titles), len(abstracts))):
        title = titles[i].strip()
        abstract = abstracts[i].strip()

        if len(abstract) > 100:  # Only use abstracts with substantial content
            full_text = f"{title}\n\n{abstract}"
            text = clean_text(full_text)
            if text:  # Only add non-empty texts
                texts.append(text)

    return texts


def extract_github_readme(response_data: Dict) -> List[str]:
    """Extract README content from GitHub API response."""
    texts = []

    # Handle error responses
    if isinstance(response_data, dict) and "status_code" in response_data:
        return []

    for repo in response_data.get("items", []):
        repo_name = repo.get("full_name", "")
        repo_description = repo.get("description", "")

        if not repo_name:
            continue

        # Get the README content
        readme_url = f"https://api.github.com/repos/{repo_name}/readme"

        try:
            rate_limiter = rate_limiters.get("github", rate_limiters["default"])
            readme_response = rate_limiter.call_with_retry(
                requests.get,
                readme_url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/vnd.github.v3+json",
                },
            )

            if readme_response.status_code == 200:
                readme_data = readme_response.json()
                # GitHub returns the content Base64 encoded
                import base64

                try:
                    content = base64.b64decode(readme_data.get("content", "")).decode(
                        "utf-8"
                    )

                    # Clean markdown and code
                    content = re.sub(r"```.*?```", " ", content, flags=re.DOTALL)
                    content = re.sub(r"`.*?`", " ", content)
                    content = re.sub(r"#{1,6}\s+", " ", content)

                    # Add the repo description as a title if available
                    if repo_description:
                        full_text = f"{repo_name}: {repo_description}\n\n{content}"
                    else:
                        full_text = f"{repo_name}\n\n{content}"

                    # Clean text
                    text = clean_text(full_text)
                    if (
                        text and len(text) > 200
                    ):  # Only add non-empty texts with sufficient content
                        texts.append(text)
                except Exception as e:
                    logger.warning(
                        f"Error processing README content for {repo_name}: {e}"
                    )
            else:
                logger.warning(
                    f"Failed to get README for {repo_name}: {readme_response.status_code}"
                )

        except Exception as e:
            logger.warning(f"Error fetching README for {repo_name}: {e}")

    return texts


def extract_gutenberg_text(response_data: bytes = None) -> List[str]:
    """Extract text from Project Gutenberg books."""
    texts = []

    # Define books to sample from Gutenberg
    GUTENBERG_TEXTS = [
        {"id": 1342, "title": "Pride and Prejudice", "author": "Jane Austen"},
        {
            "id": 11,
            "title": "Alice's Adventures in Wonderland",
            "author": "Lewis Carroll",
        },
        {"id": 84, "title": "Frankenstein", "author": "Mary Shelley"},
        {
            "id": 1661,
            "title": "The Adventures of Sherlock Holmes",
            "author": "Arthur Conan Doyle",
        },
        {"id": 2701, "title": "Moby Dick", "author": "Herman Melville"},
        {"id": 1184, "title": "The Count of Monte Cristo", "author": "Alexandre Dumas"},
        {"id": 74, "title": "The Adventures of Tom Sawyer", "author": "Mark Twain"},
        {"id": 1400, "title": "Great Expectations", "author": "Charles Dickens"},
        {"id": 2554, "title": "Crime and Punishment", "author": "Fyodor Dostoevsky"},
        {"id": 98, "title": "A Tale of Two Cities", "author": "Charles Dickens"},
        {"id": 1260, "title": "Jane Eyre", "author": "Charlotte Brontë"},
        {"id": 120, "title": "Treasure Island", "author": "Robert Louis Stevenson"},
        {
            "id": 23,
            "title": "Narrative of the Life of Frederick Douglass",
            "author": "Frederick Douglass",
        },
        {"id": 345, "title": "Dracula", "author": "Bram Stoker"},
        {"id": 4300, "title": "Ulysses", "author": "James Joyce"},
        {"id": 174, "title": "The Picture of Dorian Gray", "author": "Oscar Wilde"},
        {"id": 2814, "title": "Dubliners", "author": "James Joyce"},
        {"id": 158, "title": "Emma", "author": "Jane Austen"},
        {"id": 76, "title": "Adventures of Huckleberry Finn", "author": "Mark Twain"},
        {"id": 244, "title": "A Study in Scarlet", "author": "Arthur Conan Doyle"},
    ]

    # Limit the number of books we process to avoid rate limiting
    books_to_process = GUTENBERG_TEXTS
    random.shuffle(books_to_process)  # Randomize to get different books each time

    for book in books_to_process:
        book_id = book["id"]
        book_url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
        alternative_urls = [
            f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
            f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8",
        ]

        content = None

        try:
            rate_limiter = rate_limiters.get("default")

            # Try the primary URL first
            try:
                response = rate_limiter.call_with_retry(
                    requests.get,
                    book_url,
                    headers={"User-Agent": USER_AGENT},
                    timeout=10,
                )

                if response.status_code == 200:
                    content = response.text
            except Exception:
                pass

            # If primary URL fails, try alternatives
            if not content:
                for alt_url in alternative_urls:
                    try:
                        response = rate_limiter.call_with_retry(
                            requests.get,
                            alt_url,
                            headers={"User-Agent": USER_AGENT},
                            timeout=10,
                        )

                        if response.status_code == 200:
                            content = response.text
                            break
                    except Exception:
                        continue

            if content:
                # Find the start of the actual text (after the header)
                start_markers = [
                    "*** START OF THIS PROJECT GUTENBERG EBOOK",
                    "*** START OF THE PROJECT GUTENBERG EBOOK",
                    "*END*THE SMALL PRINT",
                    "*** START OF THE PROJECT GUTENBERG",
                    "*** START: FULL LICENSE ***",
                    "This eBook is for the use of anyone",
                ]
                start_pos = 0
                for marker in start_markers:
                    marker_pos = content.find(marker)
                    if marker_pos > 0:
                        # Find the end of this line and the next paragraph
                        eol_pos = content.find("\n", marker_pos)
                        if eol_pos > 0:
                            paragraph_pos = content.find("\n\n", eol_pos)
                            if paragraph_pos > 0:
                                start_pos = paragraph_pos + 2
                                break
                            else:
                                # If no double newline, just use end of line plus 1
                                start_pos = eol_pos + 1
                                break

                # If we couldn't find a marker, use a percentage of the file
                if start_pos == 0:
                    start_pos = int(len(content) * 0.05)  # Skip first 5%

                # Find the end of the actual text (before the footer)
                end_markers = [
                    "*** END OF THIS PROJECT GUTENBERG EBOOK",
                    "*** END OF THE PROJECT GUTENBERG EBOOK",
                    "End of Project Gutenberg's",
                    "*** END OF THIS PROJECT GUTENBERG",
                    "End of the Project Gutenberg",
                    "END OF THE PROJECT GUTENBERG",
                ]
                end_pos = len(content)
                for marker in end_markers:
                    marker_pos = content.find(marker)
                    if marker_pos > 0:
                        end_pos = marker_pos
                        break

                # If we couldn't find an end marker, use a percentage of the file
                if end_pos == len(content):
                    end_pos = int(len(content) * 0.95)  # Use up to 95%

                # Extract the actual text
                if start_pos < end_pos:
                    book_text = content[start_pos:end_pos].strip()

                    # Add book title and author
                    title = book.get("title", "")
                    author = book.get("author", "")
                    header = f"{title} by {author}\n\n"

                    # Clean text
                    full_text = header + book_text

                    # Split long texts into manageable chunks of varying sizes
                    # This creates more natural sections of text
                    chunk_sizes = [50000, 40000, 30000, 20000, 10000]

                    # If text is very large, create multiple chunks
                    if len(full_text) > min(chunk_sizes):
                        chunks = []
                        remaining_text = full_text

                        # Try to find at least 5 good chunks from this book
                        max_chunks = 5

                        for _ in range(max_chunks):
                            if len(remaining_text) < min(chunk_sizes):
                                break

                            # Choose a random size for this chunk
                            target_size = random.choice(
                                [
                                    size
                                    for size in chunk_sizes
                                    if size < len(remaining_text)
                                ]
                            )

                            # Find a good boundary to split on
                            # Look for paragraph breaks near the target size
                            ideal_pos = min(target_size, len(remaining_text))
                            split_pos = ideal_pos

                            # Try to find a paragraph break near the ideal position
                            para_search_start = max(0, ideal_pos - 1000)
                            para_search_end = min(len(remaining_text), ideal_pos + 1000)
                            para_text = remaining_text[
                                para_search_start:para_search_end
                            ]

                            # Find paragraph breaks in this region
                            para_breaks = list(re.finditer(r"\n\n+", para_text))
                            if para_breaks:
                                # Choose a paragraph break close to our target
                                best_break = min(
                                    para_breaks,
                                    key=lambda m: abs(
                                        m.start() + para_search_start - ideal_pos
                                    ),
                                )
                                split_pos = best_break.start() + para_search_start

                            # Get this chunk and clean it
                            chunk = clean_text(remaining_text[:split_pos])
                            if chunk:
                                chunks.append(chunk)

                            # Update remaining text
                            remaining_text = remaining_text[split_pos:]

                        texts.extend(chunks)
                    else:
                        # For smaller texts, just use the whole thing
                        clean_text_content = clean_text(full_text)
                        if clean_text_content:
                            texts.append(clean_text_content)
            else:
                logger.warning(f"Could not retrieve content for book {book_id}")

        except Exception as e:
            logger.warning(f"Error processing Gutenberg book {book_id}: {e}")

    return texts


# -----------------
# Text Processing
# -----------------


def clean_text(text: str) -> str:
    """Clean and normalize text for token extraction."""
    if not text:
        return ""

    # Replace multiple newlines with a single newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:\'"-]', "", text)

    # Clean up quote entities
    text = text.replace("&quot;", '"').replace("&lt;", "<").replace("&gt;", ">")

    return text.strip()


def sample_complete_text(text: str, min_length: int, max_length: int) -> str:
    """Sample a complete section of text, preserving context."""
    if not text or len(text) <= 0:
        return ""

    if len(text) <= max_length:
        return text

    # Try to find a good starting point (beginning of a paragraph or sentence)
    paragraphs = re.split(r"\n\n+", text)

    # If the text has paragraphs, try to sample complete paragraphs
    if len(paragraphs) > 1:
        # Start with a random paragraph
        start_idx = random.randint(0, len(paragraphs) - 1)
        sampled_text = ""

        # Keep adding paragraphs until we reach the max length or run out of paragraphs
        current_idx = start_idx
        while len(sampled_text) < max_length and current_idx < len(paragraphs):
            if len(sampled_text) + len(paragraphs[current_idx]) + 2 <= max_length:
                if sampled_text:
                    sampled_text += "\n\n" + paragraphs[current_idx]
                else:
                    sampled_text = paragraphs[current_idx]
                current_idx += 1
            else:
                break

        # If we have enough content, return it
        if len(sampled_text) >= min_length:
            return sampled_text

    # If we couldn't sample by paragraphs or don't have enough content,
    # try sampling by sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # Start with a random sentence
    if len(sentences) > 1:
        start_idx = random.randint(0, len(sentences) - 1)
        sampled_text = ""

        # Keep adding sentences until we reach the max length or run out of sentences
        current_idx = start_idx
        while len(sampled_text) < max_length and current_idx < len(sentences):
            if len(sampled_text) + len(sentences[current_idx]) + 1 <= max_length:
                if sampled_text:
                    sampled_text += " " + sentences[current_idx]
                else:
                    sampled_text = sentences[current_idx]
                current_idx += 1
            else:
                break

        # If we have enough content, return it
        if len(sampled_text) >= min_length:
            return sampled_text

    # If all else fails, just take a substring making sure not to cut tokens
    start_pos = random.randint(0, len(text) - max_length)

    # Adjust to start at a token boundary
    if start_pos > 0:
        while start_pos < len(text) and not text[start_pos].isspace():
            start_pos += 1
        start_pos += 1  # Skip the space

    end_pos = min(start_pos + max_length, len(text))

    # Adjust to end at a sentence boundary if possible
    sentence_end = text.rfind(".", start_pos, end_pos)
    if sentence_end > start_pos + min_length:
        end_pos = sentence_end + 1

    return text[start_pos:end_pos].strip()


# -----------------
# Data Fetching
# -----------------


def fetch_data(source: Dict, cache_dir: Path, force_refresh: bool = False) -> List[str]:
    """Fetch data from a source, using cache if available."""
    source_name = source["name"]
    cache_file = cache_dir / f"{source_name}.json"

    # Check if cache exists and is not being forcibly refreshed
    if cache_file.exists() and not force_refresh:
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                logger.info(f"Using cached data for {source_name}")
                return cached_data
        except Exception as e:
            logger.warning(f"Error reading cache for {source_name}: {e}")

    logger.info(f"Fetching data from {source_name}...")

    # Prepare request
    endpoint = source["endpoint"]
    params = source.get("params", {})
    headers = source.get("headers", {"User-Agent": USER_AGENT})

    # Get the appropriate rate limiter
    rate_limiter_name = source.get("rate_limiter", "default")
    rate_limiter = rate_limiters.get(rate_limiter_name, rate_limiters["default"])

    # Update API keys for sources that need them
    config = GeneratorConfig.load_config()

    if "opensubtitles" in source_name and "Api-Key" in headers:
        opensubtitles_key = config.corpus_generation.api_keys.get("opensubtitles")
        if opensubtitles_key and opensubtitles_key != "YOUR_OPENSUBTITLES_API_KEY":
            headers["Api-Key"] = opensubtitles_key

    # Special case for GitHub which uses Authorization header
    if "github" in source_name:
        github_key = config.corpus_generation.api_keys.get("github")
        if github_key and github_key != "YOUR_GITHUB_API_KEY":
            headers["Authorization"] = f"token {github_key}"

    if "gutenberg" in source_name or source_name == "project_gutenberg":
        # For Gutenberg, we don't need to make a request to the endpoint
        # Instead, we call the extractor directly
        extracted_texts = extract_gutenberg_text()
    else:
        # Make the request
        try:
            response = rate_limiter.call_with_retry(
                requests.get, endpoint, params=params, headers=headers, timeout=15
            )

            if response.status_code != 200:
                logger.warning(
                    f"Error fetching data from {source_name}: {response.status_code}"
                )

                # If there's an existing cache, use it even if we wanted to refresh
                if cache_file.exists():
                    try:
                        with open(cache_file, "r", encoding="utf-8") as f:
                            logger.info(
                                f"Using existing cache for {source_name} despite refresh request"
                            )
                            return json.load(f)
                    except Exception:
                        pass

                # For specific status codes like 403, add response to context
                if response.status_code in [401, 403, 429]:
                    # Pass the status code to the extraction function
                    response_data = {"status_code": response.status_code}

                    # If there's a response body, try to parse it
                    try:
                        if response.text:
                            if response.headers.get("Content-Type", "").startswith(
                                "application/json"
                            ):
                                response_data.update(response.json())
                            else:
                                response_data["text"] = response.text[
                                    :1000
                                ]  # Limit text size
                    except Exception:
                        pass

                    # Call extractor with error context
                    extract_function_name = source["extract_function"]
                    extract_function = globals()[extract_function_name]
                    extracted_texts = extract_function(response_data)

                    if extracted_texts:
                        # Cache the results even from error handling
                        os.makedirs(cache_dir, exist_ok=True)
                        with open(cache_file, "w", encoding="utf-8") as f:
                            json.dump(extracted_texts, f, ensure_ascii=False, indent=2)

                        return extracted_texts

                return []

            # Extract data based on the specified function
            extract_function_name = source["extract_function"]
            extract_function = globals()[extract_function_name]

            # Extract data using the appropriate function
            if extract_function_name in ["extract_arxiv_text"]:
                # These APIs return XML or other text formats
                extracted_texts = extract_function(response.text)
            else:
                # For APIs that return JSON
                try:
                    response_json = response.json()
                    extracted_texts = extract_function(response_json)
                except json.JSONDecodeError:
                    # If JSON parsing fails, try passing the raw text
                    extracted_texts = extract_function(
                        {"text": response.text, "status_code": response.status_code}
                    )

        except Exception as e:
            logger.error(f"Error processing data from {source_name}: {e}")

            # If there's an existing cache, use it as fallback
            if cache_file.exists():
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        logger.info(
                            f"Using cached data for {source_name} as fallback after error"
                        )
                        return json.load(f)
                except Exception:
                    pass

            return []

    # Make sure we got some text data
    if not extracted_texts:
        logger.warning(f"No text data extracted from {source_name}")

        # If there's an existing cache, use it as fallback
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    logger.info(
                        f"Using cached data for {source_name} as fallback after empty result"
                    )
                    return json.load(f)
            except Exception:
                pass

        return []

    # Cache the results
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(extracted_texts, f, ensure_ascii=False, indent=2)

    return extracted_texts


# -----------------
# Corpus Generation
# -----------------


def generate_corpus(
    config: GeneratorConfig,
    selected_categories: Optional[List[str]] = None,
    output_file: Optional[str] = None,
    force_refresh: bool = False,
    use_datasets: bool = True,
    quiet: bool = False,
) -> str:
    """Generate a corpus from selected categories."""
    # Set up directories
    output_dir = config.paths.corpuses_dir
    cache_dir = config.paths.cache_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Initialize the dataset loader if enabled
    dataset_loader = None
    if use_datasets:
        dataset_loader = DatasetLoader(cache_dir)

    # If no categories are selected, use all
    if not selected_categories:
        selected_categories = list(config.corpus_generation.categories.keys())

    # Calculate proportions
    total_proportion = sum(
        config.corpus_generation.categories.get(cat, 0) for cat in selected_categories
    )
    proportions = {
        cat: config.corpus_generation.categories.get(cat, 0) / total_proportion
        for cat in selected_categories
    }

    # Set target corpus size to 10MB by default (approximately 10M characters)
    default_corpus_size = 10 * 1024 * 1024  # 10MB

    # Calculate target size for each category
    target_corpus_size = getattr(
        config.corpus_generation, "total_corpus_size", default_corpus_size
    )

    # Adjust general category weight if others failed
    # This ensures we still get a good sized corpus even if APIs fail
    if "general" in proportions and any(
        cat not in ["general"] for cat in selected_categories
    ):
        # First check if we have enough data from other categories
        other_categories = [cat for cat in selected_categories if cat != "general"]
        other_proportion = sum(proportions[cat] for cat in other_categories)

        if (
            other_proportion < 0.5
        ):  # If we have less than 50% from non-general categories
            # Boost general to make up the difference
            proportions["general"] = max(proportions.get("general", 0), 0.5)

            # Recalculate total proportion
            total_proportion = sum(proportions.values())

            # Normalize proportions
            proportions = {
                cat: prop / total_proportion for cat, prop in proportions.items()
            }

    # Calculate target size for each category
    target_sizes = {
        cat: int(target_corpus_size * prop) for cat, prop in proportions.items()
    }

    # Prepare output
    if not output_file:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        categories_str = "_".join(selected_categories)
        output_file = output_dir / f"corpus_{categories_str}_{timestamp}.txt"
    else:
        output_file = Path(output_file)

    # Collect samples from each category
    corpus_texts = []
    corpus_size = 0

    for category in selected_categories:
        category_sources = SOURCES.get(category, [])
        category_target = target_sizes[category]
        category_texts = []

        if not quiet:
            logger.info(f"Fetching content for category '{category}'...")

        # Fetch data from each source in the category
        for source in category_sources:
            texts = fetch_data(source, cache_dir, force_refresh)
            if texts:
                category_texts.extend(texts)

        # If using datasets, try to get additional content
        if use_datasets and dataset_loader:
            dataset_texts = dataset_loader.load_dataset(category, force_refresh)
            if dataset_texts:
                if not quiet:
                    logger.info(
                        f"Added {len(dataset_texts)} samples from dataset for {category}"
                    )
                category_texts.extend(dataset_texts)

        # Sample and process texts
        category_samples = []
        category_size = 0

        # Shuffle texts
        random.shuffle(category_texts)

        # Keep track of attempts to avoid infinite loop
        attempts = 0
        max_attempts = (
            len(category_texts) * 2
        )  # Allow multiple samples from the same text

        # Extract samples until we reach the target size or run out of attempts
        while (
            category_size < category_target
            and attempts < max_attempts
            and category_texts
        ):
            # Cycle through texts
            text_index = attempts % len(category_texts)
            text = category_texts[text_index]
            attempts += 1

            # For general category (likely books), allow larger samples to preserve context
            if category == "general":
                max_length = (
                    config.corpus_generation.max_length * 10
                )  # 10x larger for books
            else:
                max_length = config.corpus_generation.max_length

            # Sample the text with complete sections
            sample = sample_complete_text(
                text,
                config.corpus_generation.min_length,
                max_length,
            )

            if sample:
                # Don't add identical samples
                if sample not in category_samples:
                    category_samples.append(sample)
                    category_size += len(sample)

        # Add category samples to corpus
        if not quiet:
            logger.info(
                f"Category '{category}': {len(category_samples)} samples, "
                f"{category_size} characters ({category_size / 1024:.1f} KB)"
            )

        corpus_texts.extend(category_samples)
        corpus_size += category_size

    # If corpus is still too small, try to get more general content
    if (
        corpus_size < target_corpus_size * 0.5
        and "general" in selected_categories
        and dataset_loader
    ):
        logger.info(
            "Corpus is smaller than expected, adding more content from datasets..."
        )

        # Try to load additional texts from datasets
        for category in selected_categories:
            additional_texts = dataset_loader.load_dataset(
                category, force_refresh=False
            )

            if additional_texts:
                # Sample more texts until we reach the target size
                additional_samples = []
                additional_size = 0
                target_additional = target_corpus_size - corpus_size

                # Shuffle the texts for variety
                random.shuffle(additional_texts)

                for text in additional_texts:
                    if additional_size >= target_additional:
                        break

                    # Use larger samples for books
                    max_length = config.corpus_generation.max_length * 10

                    # Sample the text
                    sample = sample_complete_text(
                        text,
                        config.corpus_generation.min_length,
                        max_length,
                    )

                    if sample and sample not in corpus_texts:
                        additional_samples.append(sample)
                        additional_size += len(sample)

                if not quiet and additional_samples:
                    logger.info(
                        f"Added {len(additional_samples)} additional samples from {category} dataset, "
                        f"{additional_size} characters ({additional_size / 1024:.1f} KB)"
                    )

                corpus_texts.extend(additional_samples)
                corpus_size += additional_size

                if corpus_size >= target_corpus_size * 0.8:
                    break  # Stop if we've reached 80% of target size

    # Shuffle the final corpus
    random.shuffle(corpus_texts)

    # Write corpus to file, with each text separated by a single newline
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(corpus_texts))

    if not quiet:
        logger.info(
            f"\nCorpus generated with {len(corpus_texts)} samples and "
            f"{corpus_size} characters ({corpus_size / 1024 / 1024:.2f} MB)"
        )
        logger.info(f"Saved to {output_file}")

    return str(output_file)


# -----------------
# Command Line Interface
# -----------------


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate custom corpora for chord layout optimization",
    )

    parser.add_argument(
        "-c",
        "--categories",
        help="Comma-separated list of categories to include (default: all)",
        type=str,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output file path (default: auto-generated)",
        type=str,
    )

    parser.add_argument(
        "-s",
        "--size",
        help="Target corpus size in characters (default: 10MB)",
        type=int,
    )

    parser.add_argument(
        "-m",
        "--min-length",
        help="Minimum sample length",
        type=int,
    )

    parser.add_argument(
        "-x",
        "--max-length",
        help="Maximum sample length",
        type=int,
    )

    parser.add_argument(
        "-f",
        "--force-refresh",
        help="Force refresh of cached data",
        action="store_true",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        help="Suppress progress output",
        action="store_true",
    )

    parser.add_argument(
        "--config",
        help="Path to custom config file",
        type=str,
    )

    parser.add_argument(
        "--no-datasets",
        help="Disable use of bundled datasets",
        action="store_true",
    )

    parser.add_argument(
        "--clean-cache",
        help="Clean up old cached data (older than specified days)",
        type=int,
        metavar="DAYS",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config = GeneratorConfig.load_config(args.config)

    # Override config with command line arguments
    if args.size:
        config.corpus_generation.total_corpus_size = args.size
    else:
        # Default to 10MB
        config.corpus_generation.total_corpus_size = 10 * 1024 * 1024

    if args.min_length:
        config.corpus_generation.min_length = args.min_length

    if args.max_length:
        config.corpus_generation.max_length = args.max_length

    # Clean cache if requested
    if args.clean_cache:
        cache_dir = config.paths.cache_dir
        dataset_loader = DatasetLoader(cache_dir)
        dataset_loader.cleanup_cache(args.clean_cache)
        logger.info(f"Cleaned up cache files older than {args.clean_cache} days")
        if not args.categories:  # If only cleaning cache, exit
            return

    # Parse categories
    selected_categories = None
    if args.categories:
        selected_categories = [cat.strip() for cat in args.categories.split(",")]
        # Validate categories
        valid_categories = set(config.corpus_generation.categories.keys())
        for cat in selected_categories:
            if cat not in valid_categories:
                logger.warning(
                    f"Warning: Category '{cat}' is not valid. "
                    f"Valid categories: {', '.join(valid_categories)}"
                )

        # Filter out invalid categories
        selected_categories = [
            cat for cat in selected_categories if cat in valid_categories
        ]

        if not selected_categories:
            logger.warning("No valid categories selected. Using all categories.")
            selected_categories = None

    try:
        # Generate corpus
        output_file = generate_corpus(
            config=config,
            selected_categories=selected_categories,
            output_file=args.output,
            force_refresh=args.force_refresh,
            use_datasets=not args.no_datasets,
            quiet=args.quiet,
        )

        # Verify the corpus file size
        corpus_path = Path(output_file)
        if corpus_path.exists():
            file_size = corpus_path.stat().st_size
            min_expected_size = 1 * 1024 * 1024  # At least 1MB

            if file_size < min_expected_size:
                logger.warning(
                    f"Warning: Generated corpus is only {file_size / 1024 / 1024:.2f} MB, "
                    f"which is less than the minimum expected size of {min_expected_size / 1024 / 1024:.2f} MB."
                )

                # If the corpus is too small, try again with focus on general content
                if "general" in config.corpus_generation.categories:
                    logger.info(
                        "Attempting to generate a larger corpus with more general content..."
                    )

                    # Adjust category weights to favor general
                    adjusted_categories = {"general": 0.8}

                    # Keep some other categories if they were originally selected
                    if selected_categories:
                        for cat in selected_categories:
                            if (
                                cat != "general"
                                and cat in config.corpus_generation.categories
                            ):
                                adjusted_categories[cat] = (
                                    0.2 / (len(selected_categories) - 1)
                                    if len(selected_categories) > 1
                                    else 0.2
                                )
                    else:
                        # If all categories were selected, keep a small weight for others
                        for cat in config.corpus_generation.categories:
                            if cat != "general":
                                adjusted_categories[cat] = 0.2 / (
                                    len(config.corpus_generation.categories) - 1
                                )

                    # Update config
                    config.corpus_generation.categories = adjusted_categories

                    # Generate again with adjusted weights
                    generate_corpus(
                        config=config,
                        selected_categories=list(adjusted_categories.keys()),
                        output_file=args.output,
                        force_refresh=False,  # Use cached data
                        use_datasets=not args.no_datasets,
                        quiet=args.quiet,
                    )
    except Exception as e:
        logger.error(f"Error generating corpus: {e}")
        if not args.quiet:
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
