# Optimized Chord Generator

This project implements an optimized solution for generating chord combinations for text input, taking into account word frequency distributions following Zipf's law. The goal is to minimize the average effort required for text input by assigning shorter chord combinations to more frequently used words while ensuring every word receives a valid chord assignment.

## Goal in Detail

1. Minimize the total weighted cost: $\sum_{i=1}^n f(i) * C(w_i, c_i)$
   where:

   - $f(i)$ is the Zipf frequency of word $i$
   - $C(w_i, c_i)$ is the cost function for assigning chord $c_i$ to word $w_i$

2. Adhere to the following rules:
   - Every word must receive a chord assignment
   - No chord with more than MAX-CHARS characters: $\forall c \in C: |c| \leq \text{MAX-CHARS}$
   - No chord with fewer than MIN-CHARS characters: $\forall c \in C: |c| \geq \text{MIN-CHARS}$

## Mathematical Foundation

### Cost Function

The cost function for a chord assignment is defined as:

$C(w, c) = f(r) * |c| * (2 - S(w,c)) * P(c)$

where:

- $f(r)$ is the Zipf weight for rank $r$
- $|c|$ is the length of the chord
- $S(w,c)$ is the similarity score between word and chord
- $P(c)$ is the fallback penalty (if applicable)

### Similarity Score

The similarity between a word and its chord is calculated as:

$S(w,c) = \frac{|\text{chars}(w) \cap \text{chars}(c)|}{|c|}$

This ensures that chords using characters from the original word are preferred.

### Fallback Assignment

When optimal chord assignment is not possible, the fallback system:

1. Attempts to use first/last characters of the word
2. Adds unique middle characters if needed
3. Uses character substitution as a last resort

The fallback assignments incur a penalty factor of FALLBACK_PENALTY in the cost function.

### Zipf's Distribution

The frequency weight for a word of rank $k$ in a vocabulary of size $N$ is:

$f(k;N) = \frac{1}{k H_N}$

where $H_N$ is the Nth harmonic number:

$H_N = \sum_{k=1}^N \frac{1}{k}$

## Implementation Details

### Optimization Process

1. First pass: Attempt optimal assignments using character combinations from the word
2. Fallback pass: Generate valid chords for remaining unassigned words
3. Cost calculation: Apply normalized cost function with similarity scoring

### Performance Metrics

The implementation tracks:

1. Total Weighted Cost: $C_{\text{total}} = \sum_{i} C(w_i, c_i)$
2. Approximation Ratio: $\frac{C_{\text{total}}}{C_{\text{lower}}}$
3. Character Similarity Score: $\frac{1}{n}\sum_{i} S(w_i, c_i)$
4. First/Last Character Usage Rate
5. Number of Fallback Assignments
6. Average Chord Length

## Configuration Parameters

```python
MAX_CHARS = 5            # Maximum chord length
MIN_CHARS = 2            # Minimum chord length
FALLBACK_PENALTY = 1.5   # Penalty for fallback assignments
```

## Output Format

The program generates a JSON file containing:

1. Chord assignments for all words
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
        "averageChordLength": 3.14
    },
    "chords": [...]
}
```

## Usage

```bash
python improved_generator.py
```

The program will process the input corpus and generate optimized chord assignments for all words, ensuring no words are left unassigned.
