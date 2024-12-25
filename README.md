# Optimized Chord Generator

This project implements an optimized solution for generating chord combinations for text input, taking into account word frequency distributions following Zipf's law. The goal is to minimize the average effort required for text input by assigning shorter chord combinations to more frequently used words.

## Goal in Detail

1. Minimize the total weighted cost: $\sum_{i=1}^n f(i) * C(w_i, c_i)$
   where:

   - $f(i)$ is the Zipf frequency of word $i$
   - $C(w_i, c_i)$ is the cost function for assigning chord $c_i$ to word $w_i$

2. Adhere to the following rules:
   - Single-character words do not require chord assignments
   - Every multi-character word must receive a chord assignment
   - No chord with more than MAX-CHARS characters: $\forall c \in C: |c| \leq \text{MAX-CHARS}$
   - No chord with fewer than MIN-CHARS characters: $\forall c \in C: |c| \geq \text{MIN-CHARS}$

## Mathematical Foundation

### Cost Function

The cost function for a chord assignment is defined as:

$$
C(w, c) = \begin{cases}
0 & \text{if } |w| = 1 \\
f(r) * |c| * (2 - S(w,c)) * P(c) & \text{otherwise}
\end{cases}
$$

where:

- $f(r)$ is the Zipf weight for rank $r$
- $|c|$ is the length of the chord
- $S(w,c)$ is the similarity score between word and chord
- $P(c)$ is the fallback penalty (if applicable)

### Similarity Score

The similarity between a word and its chord is calculated as:

$S(w,c) = \frac{|\text{chars}(w) \cap \text{chars}(c)|}{|c|}$

This ensures that chords using characters from the original word are preferred.

### Fallback Assignment System

When optimal chord assignment is not possible, the fallback system follows this hierarchy:

1. Single-character words: No chord assigned
2. Multi-character words:
   a. Attempts to use first/last characters of the word
   b. Adds unique middle characters if needed
   c. Uses character substitution as a last resort

The fallback assignments incur a penalty factor of FALLBACK_PENALTY in the cost function.

### Zipf's Distribution

The frequency weight for a word of rank $k$ in a vocabulary of size $N$ is:

$f(k;N) = \frac{1}{k H_N}$

where $H_N$ is the Nth harmonic number:

$H_N = \sum_{k=1}^N \frac{1}{k}$

## Implementation Details

### Optimization Process

1. Initial pass: Identify single-character words (no chord needed)
2. Main pass: Attempt optimal assignments using character combinations
3. Fallback pass: Generate valid chords for remaining unassigned words
4. Cost calculation: Apply normalized cost function with similarity scoring

### Performance Metrics

The implementation tracks:

1. Total Weighted Cost: $C_{\text{total}} = \sum_{i} C(w_i, c_i)$
2. Approximation Ratio: $\frac{C_{\text{total}}}{C_{\text{lower}}}$
3. Character Similarity Score: $\frac{1}{n}\sum_{i} S(w_i, c_i)$
4. First/Last Character Usage Rate
5. Number of Fallback Assignments
6. Average Chord Length
7. Number of Single-Character Words

## Configuration Parameters

```python
FILENAME
MAX_CHARS = 5            # Maximum chord length
MIN_CHARS = 2            # Minimum chord length
FALLBACK_PENALTY = 1.5   # Penalty for fallback assignments
```

## Output Format

The program generates a JSON file containing:

1. Chord assignments:
   - Single-character words appear as-is
   - Multi-character words appear as "word -> chord"
2. Optimization metrics
3. Performance statistics

Example output structure:

```json
{
  "name": "optimized_chords",
  "optimizationMetrics": {
    "totalCost": 245.67,
    "approximationRatio": 1.23,
    "characterSimilarity": 0.89,
    "firstLastUsage": 0.76,
    "fallbackAssignments": 42,
    "averageChordLength": 3.14,
    "singleCharWords": 5
  },
  "chords": [
    "a", // Single-character word
    "the -> th", // Multi-character word
    "and -> ad" // Multi-character word
    [...]
  ]
}
```

## Usage

```bash
python chords_generator.py
```

The program will process the input corpus and generate optimized chord assignments for all words, with special handling for single-character words.
