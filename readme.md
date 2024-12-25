# Optimized Chord Generator

This project implements an optimized solution for generating chord combinations for text input, taking into account word frequency distributions following Zipf's law. The goal is to minimize the average effort required for text input by assigning shorter chord combinations to more frequently used words.

## The Optimization Problem

### Problem Statement

Given:
- A set of words W = {w₁, w₂, ..., wₙ} ordered by frequency
- Frequencies following Zipf's law: f(k) = 1/(k * ln(1.78 + n))
- Constraints on valid chord assignments

Goal:
Minimize the total weighted cost: Σᵢ f(i) * |c(wᵢ)|
where |c(wᵢ)| is the length of the chord assigned to word wᵢ

### Constraints

1. Each chord must be 2-5 characters long
2. Chord characters must be present in the word they represent
3. Each chord must be unique
4. Each word gets at most one chord assignment

### Why This Approach Works

Our solution employs a greedy approach with frequency weighting, which provides a good approximation of the optimal solution because:

1. **Monotonic Frequency Distribution**: 
   - Zipf's law ensures that word frequencies decrease monotonically
   - This property makes greedy assignment effective

2. **Local Optimality**:
   - Assigning shortest available chords to most frequent words
   - Weighted cost consideration prevents myopic decisions

3. **Bounded Approximation**:
   - We can calculate a theoretical lower bound
   - The approximation ratio helps evaluate solution quality

## Implementation Details

The algorithm uses several optimization techniques:

1. **Weighted Combination Generation**:
   ```python
   def get_weighted_combinations(word: str, rank: int) -> List[Tuple[str, float]]:
       combos = get_valid_combinations(word)
       weight = get_word_weight(rank)
       return [(combo, len(combo) * weight) for combo in combos]
   ```

2. **Cost Calculation**:
   ```python
   def calculate_total_cost(assignments: Dict[str, str], words: List[str]) -> float:
       return sum(get_word_weight(rank) * len(chord) 
                 for rank, (word, chord) in enumerate(assignments.items()) 
                 if chord != "EMPTY")
   ```

3. **Approximation Bounds**:
   - Lower bound: Σᵢ 2 * f(i) (minimum possible chord length * frequency)
   - Actual/Lower Bound ratio indicates solution quality

## Performance Metrics

The implementation tracks several key metrics:

1. Total Weighted Cost
2. Approximation Ratio
3. Number of Unassigned Words
4. Average Chord Length

These metrics help evaluate the effectiveness of the optimization and identify potential improvements.

## Usage

```python
python chord_generator.py
```

The program will:
1. Read the input word list (sorted by frequency)
2. Generate optimized chord assignments
3. Output results with optimization metrics
4. Save assignments to a new JSON file

## Mathematical Proof Sketch

For a given set of n words, the optimal solution must satisfy:

1. **Lower Bound**: Cost ≥ Σᵢ₌₁ⁿ 2/i * ln(1.78 + i)
   - No chord can be shorter than 2 characters

2. **Greedy Choice Property**:
   - If w₁ is the most frequent word, assigning it the shortest available chord is optimal
   - Proof by contradiction: swapping a longer chord would increase total cost

3. **Approximation Guarantee**:
   - The ratio between our solution and the optimal solution is bounded
   - Bound depends on the harmonic number of vocabulary size

## Future Improvements

Potential areas for enhancement:

1. Consider ergonomic factors in chord selection
2. Implement dynamic programming for small subsets
3. Add parallel processing for large vocabularies
4. Include user-defined constraints
