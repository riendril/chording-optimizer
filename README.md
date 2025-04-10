# Chording Optimizer

This project aims to help in finding a decently optimized personal set of
chord/combo assignments to augment computer text input efficiently alongside
normal typing.

## Terminology

- token: a combination of characters (utf-8) to be triggered through the use of
  a chord
- chord/combo: a combination of keys used to trigger the output of a token
- assignment (bijective): the specified relation of a chord and its assigned
  token or a token and its assigned chord
- corpus: a set of characters/text representative of common user computer text
  input

## Functionality

- (optional) custom corpus generation
- (optional) corpus tokenization
- token selection
- chord generation
- chords to word/token assignment
- assignment improvement

### Configuration options

- layout
  - keyboard form factor
  - character layout
  - (optional) custom comfort matrix
- token selection
  - frequency ordered list of words
  - custom tokens
    - (optionally) generated from provided corpus
      - (optionally) generated from open source datasets of selected topics
- chord generation
  - number of chords
  - generation algorithm
- assignment algorithms
  - greedy assignment
  - genetic evolving
