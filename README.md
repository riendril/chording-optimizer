# Zipf Chords Generator

This project implements an optimized solution for generating chord combinations for text input on (currently only ortholinear/matrix) keyboards, taking into account word frequency distributions following Zipf's law and key positions in the chosen keyboard layout.

## Core Optimization Goals

1. Minimize the total weighted cost: $\sum_{i=1}^n f(i) * C(w_i, c_i)$
   where:
   - $f(i)$ is the Zipf frequency of word $i$
   - $C(w_i, c_i)$ is the cost function for assigning chord $c_i$ to word $w_i$

2. Optimize keyboard ergonomics:
   - Avoid same-finger usage
   - Minimize lateral stretches and scissor movements
   - Prefer home row placement where feasible

3. Maintain core rules:
   - No duplicate characters in chords
   - Single-character words exempt from chord assignment
   - Multi-character words must receive chord assignments
   - Chord length bounds: MIN_LETTERS ≤ |c| ≤ MAX_LETTERS

## Keyboard Layout Considerations

The generator is optimized for ortholinear keyboard layouts with the Canary layout as an example. Key aspects considered:

1. Finger positioning:
   - Each column maps to a specific finger
   - Vertical movement costs increase with distance
   - Adjacent key combinations

2. Movement penalties:
   - Same Finger Bigrams (SFB): Prohibited
   - Lateral Stretch Bigrams (LSB): Low penalty
   - Half Scissor Bigrams (HSB): Very low penalty
   - Full Scissor Bigrams (FSB): Medium penalty
   - Combined LSB+FSB: High penalty

## Cost Function Components

The weighted cost function incorporates:

1. Base Metrics:
   - Chord length (high weight)
   - Fallback letter usage (high weight)
   - Same finger utilization (medium weight)

2. Position Weights:
   - First letter inclusion
   - Second letter inclusion
   - Last letter inclusion

3. Layout-Specific Factors:
   - Finger travel distance
   - Row positioning
   - Hand alternation bonuses

## Generation Algorithm

1. Initial Chord Generation:
   - Generate all valid subset combinations from word letters
   - Filter against already used chords
   - Apply length constraints

2. Fallback System:
   - When no valid subset exists:
     a. Identify most compatible additional letter based on layout
     b. Generate new subset combinations
     c. Recurse if needed

3. Cost Evaluation:
   - Apply weighted cost function to valid candidates
   - Select optimal chord based on combined metrics

## Configuration

The system uses a config file (generator.config) for customization:

```ini
KEYLAYOUT_TYPE = ortholinear
MAX_LETTERS = 6
MIN_LETTERS = 2

# Cost Weights
CHORD_LENGTH_WEIGHT = 1.8
FALLBACK_PENALTY = 1.8
FIRST_LETTER_WEIGHT = 1.5
SECOND_LETTER_WEIGHT = 1.2
LAST_LETTER_WEIGHT = 1.2
```

## Output Format

The generator produces a JSON file containing:

1. Chord assignments
2. Optimization metrics:
   - Total weighted cost
   - Approximation ratio
   - First/last letter usage rate
   - Fallback assignment count
   - Average chord length
   - Single letter word count

## Usage

```bash
python chords_generator.py [corpus_file] [--config path/to/config]
```

The program processes the input corpus and generates optimized chord assignments considering both frequency distribution and keyboard ergonomics.
