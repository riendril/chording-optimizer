# Chording Optimizer

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MPL--2.0-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A tool for optimizing chord/combo assignments to enhance text input efficiency alongside traditional typing. The optimizer analyzes text corpora to find optimal mappings between keyboard chord combinations and text tokens based on usage patterns and ergonomic considerations.

## 🎯 Overview

The optimizer analyzes text corpora to identify frequently used character sequences and maps them to ergonomic keyboard chord combinations. This enables faster text input by replacing common words, phrases, or character sequences with single chord presses.

## ✨ Features

- **📊 Corpus Analysis**: Processes text corpora to identify token candidates
- **🎹 Keyboard Layout Support**: Supports multiple keyboard layouts with customizable comfort matrices
- **🧠 Token Selection**: Uses iterative algorithm considering usage frequency and typing efficiency
- **⚡ Chord Generation**: Generates all valid chord combinations within specified constraints
- **🎯 Assignment Optimization**: Implements greedy and genetic algorithms for token-chord mapping
- **📈 Performance Benchmarking**: Includes performance tracking and analysis
- **🔧 Configurable**: Features YAML-based configuration system

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Required Python packages (see `requirements.txt`)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/chording-optimizer.git
cd chording-optimizer

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

1. **Configure the optimizer**: Edit `data/config.yaml` to specify your preferences:

   - Choose your keyboard layout file
   - Select your text corpus
   - Set optimization parameters
   - Choose which pipeline stages to run

2. **Run the optimization pipeline**:

   ```bash
   python chording-optimizer.py
   ```

3. **View results**: Check the `data/results/` directory for your optimized chord assignments.

## 📖 Documentation

For detailed information about the concepts and file formats used by this optimizer, see:

- **[Chording Concepts](docs/chording-concepts.md)**: Learn about chording alongside typing and how it enhances text input efficiency
- **[File Formats](docs/file-formats.md)**: Understanding corpus files, token files, and keyboard layout files

### Configuration

The optimizer is configured via `data/config.yaml`. This file contains detailed comments explaining all available options. Key configuration areas include:

- **Active Files**: Specify which keyboard layout and corpus to use
- **Pipeline Stages**: Choose which optimization steps to run
- **Algorithm Parameters**: Tune the optimization algorithms
- **Output Settings**: Control debugging and benchmarking options

### Pipeline Stages

1. **Corpus Generation** (optional): Generate custom text corpora from various sources
2. **Token Extraction**: Analyze corpus to identify and score potential tokens
3. **Chord Generation**: Generate all valid chord combinations for the keyboard layout
4. **Assignment**: Optimize token-chord mappings using selected algorithm
5. **Analysis**: Evaluate and visualize the final assignments

### Algorithms

#### Token Selection

- **Iterative Selection**: Uses dynamic programming to optimally segment text using currently selected tokens
- **Frequency Analysis**: Considers token usage frequency and replacement potential
- **Type-aware Selection**: Supports different token types (characters, words, n-grams)

#### Assignment Optimization

- **Greedy Algorithm**: Provides fast, deterministic assignment based on cost functions
- **Genetic Algorithm**: Uses evolutionary approach for global optimization

## 📁 Project Structure

```
chording-optimizer/
├── src/
│   ├── common/           # Shared utilities and types
│   ├── token_generation/ # Token extraction and selection
│   ├── chord_generation/ # Chord generation algorithms
│   ├── assignment/       # Token-chord assignment optimization
│   └── analyzing/        # Analysis and visualization tools
├── data/
│   ├── config.yaml       # Main configuration file
│   ├── keyLayouts/       # Keyboard layout definitions
│   ├── corpuses/         # Text corpora for analysis
│   ├── tokens/           # Generated token collections
│   ├── chords/           # Generated chord collections
│   └── results/          # Final optimization results
└── chording-optimizer.py # Main entry point
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the Mozilla Public License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the stenography and keyboard optimization communities
- Special thanks to:
  - **CharaChorder** for giving me the idea of chording alongside typing
  - **r/KeyboardLayouts (AKL)** subreddit community for keyboard optimization discussions
  - **Imprint Discord** (Digital Taylor) for ergonomic keyboard design insights
  - **empressabyss** for the Nordrassil ergonomic keyboard layout

## 📞 Support

- 📧 Create an issue on GitHub for bug reports or feature requests
- 💬 Use GitHub Discussions for questions and community support
- 📖 Check the documentation for detailed usage instructions
