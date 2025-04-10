"""
Tokenizer for chord system optimization.

This module analyzes text to identify tokens (characters, character n-grams,
words, etc.) that would benefit most from chord assignments.
"""

import argparse
import multiprocessing
import re
import sys
from collections import Counter
from functools import partial, reduce
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import yaml

from src.common.config import GeneratorConfig
from src.common.shared_types import Finger, TokenCollection, WordData

# -----------------
# Text Preprocessing
# -----------------


def preprocess_text(text: str) -> str:
    """Clean and normalize text for token extraction."""
    return re.sub(r"\s+", " ", text.lower()).strip()


# -----------------
# Token Extraction
# -----------------


def character_tokens(text: str) -> Dict[str, int]:
    """Extract individual characters and their frequencies."""
    return Counter(text)


def character_ngrams(text: str, n: int) -> Dict[str, int]:
    """Extract character n-grams and their frequencies."""
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def words(text: str) -> Dict[str, int]:
    """Extract words (no spaces) and their frequencies."""
    return Counter(re.findall(r"\b[\w\']+\b", text))


def words_with_space(text: str) -> Dict[str, int]:
    """Extract words with trailing space and their frequencies."""
    return Counter(re.findall(r"\b[\w\']+\s", text))


def word_ngrams(text: str, n: int) -> Dict[str, int]:
    """Extract word n-grams and their frequencies."""
    word_list = re.findall(r"\b[\w\']+\b", text)
    return Counter(
        " ".join(word_list[i : i + n]) for i in range(len(word_list) - n + 1)
    )


def punctuation_patterns(text: str) -> Dict[str, int]:
    """Extract common punctuation patterns."""
    patterns = [
        r"\.{2,}",  # multiple periods
        r"[,\.;:][\'\"]",  # punctuation followed by quotes
        r"\w+\.\w+",  # domain-like patterns
        r"[!?]{2,}",  # multiple ! or ?
    ]

    extract_pattern = lambda pattern: re.findall(pattern, text)
    all_matches = [extract_pattern(pattern) for pattern in patterns]
    flattened = [item for sublist in all_matches for item in sublist]

    return Counter(flattened)


def merge_counters(counters: List[Counter]) -> Counter:
    """Merge multiple counters into one."""
    return reduce(lambda x, y: x + y, counters, Counter())


def extract_tokens(text: str, config: GeneratorConfig) -> Dict[str, int]:
    """Extract tokens from text using functional composition."""
    processed_text = preprocess_text(text)
    token_config = config.token_analysis

    extractors = []

    # Add character tokens if enabled
    if token_config.include_characters:
        extractors.append(partial(character_tokens, processed_text))

    # Add character n-grams if enabled
    if token_config.include_character_ngrams:
        min_n = max(2, token_config.min_token_length)
        max_n = min(5, token_config.max_token_length)  # Cap at 5 for character n-grams
        extractors.extend(
            [
                partial(character_ngrams, processed_text, n)
                for n in range(min_n, max_n + 1)
            ]
        )

    # Add words if enabled
    if token_config.include_words:
        extractors.append(partial(words, processed_text))
        extractors.append(partial(words_with_space, processed_text))

    # Add word n-grams if enabled
    if token_config.include_word_ngrams:
        min_n = max(2, token_config.min_token_length)
        max_n = min(4, token_config.max_token_length)  # Cap at 4 for word n-grams
        extractors.extend(
            [partial(word_ngrams, processed_text, n) for n in range(min_n, max_n + 1)]
        )

    # Add punctuation patterns
    extractors.append(partial(punctuation_patterns, processed_text))

    # Apply all extractors and merge results
    return merge_counters([extractor() for extractor in extractors])


# ------------------
# Difficulty Scoring
# ------------------


def key_difficulty(char: str, layout_config: Dict) -> float:
    """Get difficulty value for a single key."""
    comfort_values = layout_config["comfort"]
    return comfort_values.get(char, 10)  # Default to 10 for unknown keys


def base_difficulty(token: str, layout_config: Dict) -> float:
    """Calculate cumulative base difficulty of all keys in token."""
    return sum(key_difficulty(char, layout_config) for char in token)


def get_finger(char: str, layout_config: Dict) -> Optional[Finger]:
    """Get the finger used for a character."""
    finger_map = layout_config["fingers"]
    finger_name = finger_map.get(char)
    return Finger[finger_name] if finger_name else None


def same_finger(a: str, b: str, layout_config: Dict) -> bool:
    """Check if two characters are typed with the same finger."""
    finger_a = get_finger(a, layout_config)
    finger_b = get_finger(b, layout_config)
    return finger_a is not None and finger_b is not None and finger_a == finger_b


def key_distance(a: str, b: str, layout_config: Dict) -> float:
    """Calculate physical distance between two keys."""
    positions = layout_config["positions"]
    if a not in positions or b not in positions:
        return 0

    a_pos, b_pos = positions[a], positions[b]
    return ((a_pos[0] - b_pos[0]) ** 2 + (a_pos[1] - b_pos[1]) ** 2) ** 0.5


def transition_difficulty(
    token: str, layout_config: Dict, prev_token: str = "", next_token: str = ""
) -> float:
    """
    Calculate difficulty of transitions between adjacent keys.
    """
    if not token:
        return 0

    difficulty = 0

    # Check transition from previous token's last character to this token's first
    if prev_token and token:
        prev_last = prev_token[-1]
        curr_first = token[0]

        if same_finger(prev_last, curr_first, layout_config):
            difficulty += key_distance(prev_last, curr_first, layout_config)

    # Check transitions within the current token
    for i in range(len(token) - 1):
        a, b = token[i], token[i + 1]
        if same_finger(a, b, layout_config):
            difficulty += key_distance(a, b, layout_config)

    # Check transition from this token's last character to next token's first
    if token and next_token:
        curr_last = token[-1]
        next_first = next_token[0]

        if same_finger(curr_last, next_first, layout_config):
            difficulty += key_distance(curr_last, next_first, layout_config)

    return difficulty


def typing_difficulty(
    token: str, layout_config: Dict, prev_token: str = "", next_token: str = ""
) -> float:
    """Calculate overall typing difficulty score."""
    return base_difficulty(token, layout_config) + transition_difficulty(
        token, layout_config, prev_token, next_token
    )


# ------------------
# Token Scoring
# ------------------


def score_token(
    token_freq_tuple: Tuple[str, int],
    layout_config: Dict,
    context_tokens: Dict[str, str] = None,
) -> Dict:
    """
    Score a token based on frequency and typing difficulty.
    """
    token, frequency = token_freq_tuple

    # Get context tokens if available
    prev_token = context_tokens.get("prev", "") if context_tokens else ""
    next_token = context_tokens.get("next", "") if context_tokens else ""

    # Calculate difficulty with context if available
    difficulty = typing_difficulty(token, layout_config, prev_token, next_token)
    length = len(token)

    # Length benefit is non-linear
    length_benefit = length**1.5

    # Final score: higher makes it more attractive for chording
    score = frequency * length_benefit * (difficulty + 1)

    return {
        "token": token,
        "frequency": frequency,
        "length": length,
        "difficulty": difficulty,
        "score": score,
    }


def sort_by_score(tokens: List[Dict], reverse: bool = True) -> List[Dict]:
    """Sort tokens by their score."""
    return sorted(tokens, key=lambda x: x["score"], reverse=reverse)


def take_top_n(tokens: List[Dict], n: int) -> List[Dict]:
    """Take top n tokens from sorted list."""
    return tokens[:n]


# ------------------
# Program Composition
# ------------------


def analyze_corpus(
    corpus: str,
    config: GeneratorConfig,
    top_n: Optional[int] = None,
    show_progress: bool = True,
) -> List[Dict]:
    """Analyze corpus and return top n tokens ranked by score."""
    # Get keyboard layout configuration
    from src.common.layout import load_keyboard_layout

    layout_path = config.paths.get_layout_path(config.active_layout)
    layout_config = load_keyboard_layout(layout_path)

    if top_n is None:
        top_n = config.token_analysis.top_n_tokens

    if show_progress:
        print(f"Extracting tokens from corpus using {layout_config['name']} layout...")

    # Extract tokens
    tokens_with_freq = extract_tokens(corpus, config).items()
    tokens_list = list(tokens_with_freq)

    if show_progress:
        total_tokens = len(tokens_list)
        print(f"Found {total_tokens} unique tokens. Scoring tokens...")

    # Score tokens (in parallel if requested)
    score_with_layout = partial(score_token, layout_config=layout_config)

    if config.token_analysis.use_parallel_processing and total_tokens > 1000:
        try:
            # Determine optimal chunk size based on number of cores
            num_cores = multiprocessing.cpu_count()
            chunk_size = max(1, total_tokens // (num_cores * 4))

            if show_progress:
                print(f"Parallel processing with {num_cores} cores...")

            # Create a pool and process tokens in parallel
            with multiprocessing.Pool(processes=num_cores) as pool:
                scored_tokens = pool.map(score_with_layout, tokens_list, chunk_size)
        except Exception as e:
            if show_progress:
                print(
                    f"Parallel processing failed: {e}. Falling back to sequential processing."
                )
            # Fall back to sequential processing
            scored_tokens = process_tokens_sequentially(
                tokens_list, score_with_layout, show_progress
            )
    else:
        # Process sequentially with progress updates
        scored_tokens = process_tokens_sequentially(
            tokens_list, score_with_layout, show_progress
        )

    if show_progress:
        print("Sorting tokens by score...")

    # Sort and take top n
    return take_top_n(sort_by_score(scored_tokens), top_n)


def process_tokens_sequentially(
    tokens_list: List[Tuple[str, int]],
    scoring_func: Callable,
    show_progress: bool = True,
) -> List[Dict]:
    """Process tokens sequentially with optional progress tracking."""
    total = len(tokens_list)
    results = []

    for i, token_freq in enumerate(tokens_list):
        # Update progress every 100 tokens
        if show_progress and i % max(1, total // 100) == 0:
            progress = (i / total) * 100
            print(
                f"  Progress: {progress:.1f}% ({i}/{total} tokens processed)", end="\r"
            )

        # Score the token
        results.append(scoring_func(token_freq))

    if show_progress:
        print("  Progress: 100.0% ({}/{} tokens processed)".format(total, total))

    return results


# ------------------
# Conversion Functions
# ------------------


def create_token_collection(
    scored_tokens: List[Dict], name: str, source: Optional[str] = None
) -> TokenCollection:
    """Convert scored tokens to a TokenCollection."""
    tokens = []

    for i, token_data in enumerate(scored_tokens):
        word_data = WordData.from_word(
            word=token_data["token"],
            frequency=token_data["frequency"],
            rank=i,
            zipf_weight=1.0 / (i + 1),  # Simple Zipf approximation
            score=token_data["score"],
            difficulty=token_data["difficulty"],
        )
        tokens.append(word_data)

    return TokenCollection(
        name=name, tokens=tokens, ordered_by_frequency=True, source=source
    )


# ------------------
# Utility Functions
# ------------------


def format_token_result(index: int, token_data: Dict) -> str:
    """Format token result for display."""
    return (
        f"{index+1}. Token: '{token_data['token']}' | "
        f"Score: {token_data['score']:.2f} | "
        f"Freq: {token_data['frequency']} | "
        f"Difficulty: {token_data['difficulty']:.2f}"
    )


def print_results(results: List[Dict]) -> None:
    """Print formatted token results."""
    for i, result in enumerate(results):
        print(format_token_result(i, result))


def read_corpus_from_file(file_path: str) -> str:
    """Read corpus data from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def print_help():
    """Print detailed help information about the program."""
    help_text = """
Token Analyzer for Chording Keyboard Layout Optimization
--------------------------------------------------------

DESCRIPTION:
    This program analyzes text to identify tokens (characters, character n-grams, 
    words, word n-grams) that would benefit most from chording on a custom keyboard layout.
    It scores tokens based on typing difficulty, length, and frequency in the input text.

USAGE:
    python -m src.corpus_tokenization.token_analyzer [corpus_file] [options]

ARGUMENTS:
    corpus_file            Path to a text file to analyze (optional, uses default from config)

OPTIONS:
    -n, --top_n N         Number of top tokens to display (default: from config)
    -p, --parallel        Use parallel processing for large token sets (default: from config)
    -o, --output FILE     Save results to specified file
    -q, --quiet           Suppress progress output
    -c, --config FILE     Path to custom config file
    -h, --help            Display this help message and exit

EXAMPLES:
    # Analyze a text file and show top 50 tokens
    python -m src.corpus_tokenization.token_analyzer data/corpuses/brown.txt -n 50

    # Analyze with custom config and save to file
    python -m src.corpus_tokenization.token_analyzer -c my_config.yaml -o results.json

    # Run quietly with default corpus
    python -m src.corpus_tokenization.token_analyzer --quiet

OUTPUT FORMAT:
    For each token, the program displays:
    - Token: The actual text string
    - Score: Combined metric of frequency and typing efficiency
    - Freq: Number of occurrences in the corpus
    - Difficulty: Typing difficulty score (lower is easier)

ABOUT SCORING:
    The score is calculated based on:
    - Token frequency (more frequent tokens score higher)
    - Token length (longer tokens get a non-linear bonus)
    - Typing difficulty (based on key positions and finger usage)
    
    Higher scores indicate tokens that would benefit more from chording.
"""
    print(help_text)


# ------------------
# Main
# ------------------


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Analyze tokens from a text corpus for chording efficiency",
        add_help=False,  # Disable default help to use our custom help
    )
    parser.add_argument("corpus_file", nargs="?", help="Path to the corpus file")
    parser.add_argument(
        "-n", "--top_n", type=int, help="Number of top tokens to return"
    )
    parser.add_argument(
        "-p",
        "--parallel",
        action="store_true",
        help="Use parallel processing for large token sets",
    )
    parser.add_argument("-o", "--output", help="Output file to save results")
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress progress output"
    )
    parser.add_argument("-c", "--config", help="Path to custom config file")
    parser.add_argument(
        "-h", "--help", action="store_true", help="Show detailed help information"
    )

    # Check for help flag first
    if "-h" in sys.argv or "--help" in sys.argv:
        print_help()
        sys.exit(0)

    # Parse arguments
    args = parser.parse_args()
    show_progress = not args.quiet

    # Load configuration
    config_path = args.config if args.config else None
    config = GeneratorConfig.load_config(config_path)

    # Override config with command line arguments if provided
    if args.parallel is not None:
        config.token_analysis.use_parallel_processing = args.parallel

    top_n = args.top_n if args.top_n else config.token_analysis.top_n_tokens

    # Determine corpus to use
    if args.corpus_file:
        try:
            corpus_path = Path(args.corpus_file)
            corpus = read_corpus_from_file(corpus_path)
            if show_progress:
                print(f"Analyzing corpus from {corpus_path}...")

            # Update active corpus in config if it's in the expected directory
            if corpus_path.parent.name == config.paths.corpuses_dir.name:
                config.active_corpus = corpus_path.stem
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
    else:
        # Use default corpus from config
        try:
            corpus_path = config.paths.get_corpus_path(config.active_corpus)
            if show_progress:
                print(f"Using default corpus from {corpus_path}...")
            corpus = read_corpus_from_file(corpus_path)
        except Exception as e:
            print(f"Error reading default corpus: {e}")
            print("Using sample text instead...")
            corpus = """
            The quick brown fox jumps over the lazy dog. The dog was not very happy about this.
            What the fox was thinking, nobody knows. It wasn't the first time this had happened,
            and it probably wouldn't be the last time either. The quick brown fox jumps over
            the lazy dog again. What the heck is going on with these animals?
            """

    # Analyze corpus
    top_tokens = analyze_corpus(
        corpus, config, top_n=top_n, show_progress=show_progress
    )

    # Create token collection
    token_collection = create_token_collection(
        top_tokens,
        name=f"tokens_{config.active_corpus}_{top_n}",
        source=str(corpus_path) if "corpus_path" in locals() else "sample_text",
    )

    # Print results
    if show_progress:
        print(f"\nTop {len(top_tokens)} tokens by score:")
        print("-" * 60)
    print_results(top_tokens)

    # Output to a file if requested
    if args.output:
        try:
            output_path = Path(args.output)
            token_collection.save_to_file(output_path)
            if show_progress:
                print(f"\nResults saved to {output_path}")
        except Exception as e:
            print(f"Error writing to output file: {e}")

    # If no output file specified but not quiet, save to default location
    elif not args.quiet:
        try:
            output_filename = f"{config.active_corpus}_tokens_{top_n}.json"
            output_path = config.paths.tokens_dir / output_filename
            token_collection.save_to_file(output_path)
            print(f"\nResults saved to {output_path}")
        except Exception as e:
            print(f"Error writing to default output file: {e}")


if __name__ == "__main__":
    main()
