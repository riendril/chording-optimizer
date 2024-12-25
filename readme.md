# Optimized Chord Generator

This project implements an optimized solution for generating chord combinations for text input, taking into account word frequency distributions following Zipf's law. The goal is to minimize the average effort required for text input by assigning shorter chord combinations to more frequently used words while adhering to specific optimization constraints.

## Goal in Detail

1. Minimize the total weighted cost: $\sum_{i=1}^n f(i) * |c(w_i)|$
   where:

   - $f(i)$ is the frequency of word $i$
   - $|c(w_i)|$ is the length of the chord assigned to word $i$

2. Adhere to the following rules:

   - No chord with more than MAX_CHARS characters: $\forall c \in C: |c| \leq \text{MAX\_CHARS}$

3. Balance between the following optimization goals (with weights):

   - First and last character preference: $w_{\text{fl}} * \sum_{i=1}^n \text{has\_first\_last}(c_i, w_i)$
   - Character similarity: $w_{\text{sim}} * \sum_{i=1}^n \text{different\_chars}(c_i, w_i)$

   where:

   - $w_{\text{fl}} = \text{WEIGHT\_FIRST\_LAST\_CHAR}$
   - $w_{\text{sim}} = \text{WEIGHT\_NO\_DIFFERENT\_CHARS}$

## Mathematical Foundation

### Zipf's Distribution

For a vocabulary of size $N$, the Zipf distribution assigns to the element of rank $k$ the probability:

$f(k;N) = \begin{cases}
    \frac{1}{H_N} \frac{1}{k}, & \text{if } 1 \leq k \leq N \\
    0, & \text{if } k < 1 \text{ or } N < k
\end{cases}$

where $H_N$ is the Nth harmonic number:

$H_N = \sum_{k=1}^N \frac{1}{k}$

### Optimization Problem

The complete optimization problem can be formulated as:

Minimize:
$\sum_{i=1}^n \left(\frac{|c(w_i)|}{i H_n}\right) - w_{\text{fl}} * B_{\text{fl}}(w_i, c_i) + w_{\text{sim}} * D(w_i, c_i)$

where:

- $B_{\text{fl}}(w_i, c_i)$ is the bonus for including first/last characters
- $D(w_i, c_i)$ is the penalty for different characters

Subject to:

1. $\forall i: 2 \leq |c(w_i)| \leq \text{MAX\_CHARS}$
2. $\forall i,j: i \neq j \implies c(w_i) \neq c(w_j)$
3. $\forall i: c(w_i) \subseteq \text{chars}(w_i)$

## Implementation Details

### Cost Function

The total cost for a chord assignment is calculated as:

```python
def calculate_total_cost(assignments: Dict[str, str], words: List[str]) -> float:
    return sum(
        (len(chord) / (i + 1) / harmonic(len(words))) -
        (WEIGHT_FIRST_LAST_CHAR * has_first_last_bonus(word, chord)) +
        (WEIGHT_NO_DIFFERENT_CHARS * different_chars_penalty(word, chord))
        for i, (word, chord) in enumerate(assignments.items())
        if chord != "EMPTY"
    )
```

### Optimization Weights

The balance between different optimization goals is controlled by:

```python
WEIGHT_FIRST_LAST_CHAR = 0.3  # Bonus for including first/last chars
WEIGHT_NO_DIFFERENT_CHARS = 0.5  # Penalty for different chars
```

### Performance Metrics

The implementation tracks:

1. Total Weighted Cost: $C_{\text{total}} = \sum_{i} \text{cost}(w_i, c_i)$
2. Approximation Ratio: $\frac{C_{\text{total}}}{C_{\text{lower}}}$
3. Character Similarity Score: $\frac{1}{n}\sum_{i} \text{similarity}(w_i, c_i)$
4. First/Last Character Usage: $\frac{|\{i: \text{has\_first\_last}(w_i, c_i)\}|}{n}$

## Theoretical Bounds

### Lower Bound

The theoretical lower bound for the total cost is:

$C_{\text{lower}} = \sum_{i=1}^n \frac{2}{i H_n}$

where 2 is the minimum chord length.

### Upper Bound

The worst-case upper bound is:

$C_{\text{upper}} = \sum_{i=1}^n \frac{\text{MAX\_CHARS}}{i H_n}$

## Usage

```python
python chord_generator.py
```

Configuration parameters can be adjusted in the code:

```python
MAX_CHARS = 5
WEIGHT_FIRST_LAST_CHAR = 0.3
WEIGHT_NO_DIFFERENT_CHARS = 0.5
```

## Output Format

The program generates a JSON file with:

1. Chord assignments
2. Optimization metrics
3. Performance statistics

Example:

```json
{
    "name": "optimized_chords",
    "optimizationMetrics": {
        "totalCost": 245.67,
        "approximationRatio": 1.23,
        "characterSimilarity": 0.89,
        "firstLastUsage": 0.76
    },
    "chords": [...]
}
```
